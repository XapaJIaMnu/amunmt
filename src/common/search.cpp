#include "common/search.h"

#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"
#include "common/util/file_stream.hh"
#include "cpu/decoder/encoder_decoder.h"

using namespace std;

namespace amunmt {

Search::Search(const God &god)
{
  deviceInfo_ = god.GetNextDevice();
  scorers_ = god.GetScorers(deviceInfo_);
  bestHyps_ = god.GetBestHyps(deviceInfo_);
}

Search::~Search()
{
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif
}
  
States Search::NewStates() const
{
  size_t numScorers = scorers_.size();

  States states(numScorers);
  for (size_t i = 0; i < numScorers; i++) {
    Scorer &scorer = *scorers_[i];
    states[i].reset(scorer.NewState());
  }

  return states;
}

size_t Search::MakeFilter(const God &god, const std::set<Word>& srcWords, size_t vocabSize) {
  filterIndices_ = god.GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

void Search::Encode(const Sentences& sentences, States& states) {
  for (size_t i = 0; i < scorers_.size(); i++) {
    Scorer &scorer = *scorers_[i];
    scorer.SetSource(sentences);

    scorer.BeginSentenceState(*states[i], sentences.size());
  }
}

std::string Search::GetStringFromHypo(HypothesisPtr hypo) {
  std::vector<std::string> words;
  while (hypo) {
    words.push_back(std::to_string(hypo->GetWord()));
    hypo = hypo->GetPrevHyp();
  }
  std::string out = "";
  for (size_t i = words.size(); i > 0; i--) {
    out += words[i-1];
    if (i > 1) {
      out += ",";
    }
  }
  return out;
}

std::string Search::GetStringsFromStates(State& state) {
  std::string out = "";
  auto& stateMatrix = state.get<CPU::EncoderDecoderState>().GetStates();
  for (size_t i = 0; i < stateMatrix.rows(); i++) {
    for (size_t j = 0; j < stateMatrix.columns(); j++) {
      out += std::to_string(stateMatrix(i,j)) + " ";
    }
    out += '\n';
  }
  return out;
}

void Search::Decode(
		const God &god,
		const Sentences& sentences,
		const States &states,
		States &nextStates,
		std::shared_ptr<Histories> &histories,
		Beam &prevHyps)
{
  size_t batchSize = sentences.size();

  std::vector<size_t> beamSizes(batchSize, 1);

  // Create the files to drop the states
  util::scoped_fd fd1;
  util::FileStream hyposFile;
  fd1.reset(util::CreateOrThrow("dropStates/hypotheses.txt"));
  hyposFile.SetFD(fd1.get());

  util::scoped_fd fd2;
  util::FileStream statesFile;
  fd2.reset(util::CreateOrThrow("dropStates/states.txt"));
  statesFile.SetFD(fd2.get());

  util::scoped_fd fd3;
  util::FileStream finishedFile;
  fd3.reset(util::CreateOrThrow("dropStates/finished.txt"));
  finishedFile.SetFD(fd3.get());

  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {

    // Dropping the hypotheses themselves
    for (size_t h = 0; h < prevHyps.size(); h++) {
      hyposFile << GetStringFromHypo(prevHyps[h]) << '\n';
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Decode(god, state, nextState, beamSizes);
      // after this step, the nextState is populated
    }

    // Dropping the states
    statesFile << GetStringsFromStates(*nextStates[0]);

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
      beamSize = god.Get<size_t>("beam-size");
      }
    }
    Beams beams(batchSize);
    bool returnAlignment = god.Get<bool>("return-alignment");

    bestHyps_->CalcBeam(god, prevHyps, scorers_, filterIndices_, returnAlignment, beams, beamSizes);

    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        histories->at(i)->Add(beams[i], histories->at(i)->size() == 3 * sentences.at(i)->GetWords().size());
      }
    }

    Beam survivors;
    for (size_t batchID = 0; batchID < batchSize; ++batchID) {
      for (auto& h : beams[batchID]) {
        if (h->GetWord() != EOS) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchID];

          // Print the finished sentence
          finishedFile << GetStringFromHypo(h) << " ||| " << h->GetCost() << '\n';
        }
      }
    }

    if (survivors.size() == 0) {
      break;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
  }
}

std::shared_ptr<Histories> Search::Process(const God &god, const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  std::shared_ptr<Histories> histories(new Histories(god, sentences));

  size_t batchSize = sentences.size();
  size_t numScorers = scorers_.size();

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));

  States states = NewStates();
  States nextStates = NewStates();

  // calc
  PreProcess(god, sentences, histories, prevHyps);
  Encode(sentences, states);
  Decode(god, sentences, states, nextStates, histories, prevHyps);
  PostProcess();

  LOG(progress) << "Batch " << sentences.GetTaskCounter() << "." << sentences.GetBunchId()
                << ": Search took " << timer.format(3, "%ws");

  return histories;
}

void Search::PreProcess(
		const God &god,
		const Sentences& sentences,
		std::shared_ptr<Histories> &histories,
		Beam &prevHyps)
{
  size_t vocabSize = scorers_[0]->GetVocabSize();

  for (size_t i = 0; i < histories->size(); ++i) {
    History &history = *histories->at(i).get();
    history.Add(prevHyps);
  }

  bool filter = god.Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    std::set<Word> srcWords;
    for (size_t i = 0; i < sentences.size(); ++i) {
      const Sentence &sentence = *sentences.at(i);
      for (const auto& srcWord : sentence.GetWords()) {
        srcWords.insert(srcWord);
      }
    }
    vocabSize = MakeFilter(god, srcWords, vocabSize);
  }

}

void Search::PostProcess()
{
  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }
}


}

