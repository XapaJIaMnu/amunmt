#include "common/search.h"

#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"
#include "common/util/file_stream.hh"
#include "cpu/decoder/encoder_decoder.h"
#include "cpu/mblas/matrix.h"
#include "cpu/decoder/best_hyps.h"

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
  }
  return out;
}

std::string Search::GetStringsFromStates_presoftmax(std::vector<CPU::mblas::Matrix>& state_matrices, std::vector<std::pair<size_t, size_t> >& positions,
 std::vector<std::string>& prevHypsStr) {
  CPU::mblas::Matrix& T = state_matrices[0];
  CPU::mblas::Matrix& W4 = state_matrices[1];
  std::string out = "";
  
  //What we do here is concatinate the decoder state before going to the output layer (T) with the output layer weights (W4) at the positions
  //at the rows of T corresponding to the chosen beams according to positions and at the cols of W4 corresponding to the chosen words for those
  //beams.

  for (size_t pos = 0; pos < positions.size(); pos++) {
    size_t beamIDX = positions[pos].first;
    size_t wordIDX = positions[pos].second;
    out += prevHypsStr[pos] + " ||| ";
    for (size_t i = 0; i < W4.rows(); i++) {
      out += std::to_string(W4(i,wordIDX)) + " ";
    }
    for (size_t i = 0; i < T.columns(); i++) {
      out += std::to_string(T(beamIDX,i)) + " ";
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
  util::scoped_fd fd2;
  util::FileStream statesFile;
  const auto filename2 = "dropStates/states_" + std::to_string(sentences.at(0)->GetLineNum()) + ".txt";
  fd2.reset(util::CreateOrThrow(filename2.data()));
  statesFile.SetFD(fd2.get());

  util::scoped_fd fd3;
  util::FileStream finishedFile;
  const auto filename3 = "dropStates/finished_" + std::to_string(sentences.at(0)->GetLineNum()) + ".txt";
  fd3.reset(util::CreateOrThrow(filename3.data()));
  finishedFile.SetFD(fd3.get());

  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {

    //This gives up a copy of the matrices before they multiplied to get the output layer
    //the goal is to pair concatinate them together for the chosen vocabulary items
    //and use that as a NN input for training a regression model.
    std::vector<std::vector<CPU::mblas::Matrix> > preOutputStates(scorers_.size());
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Decode_States(god, state, nextState, beamSizes, preOutputStates[i]);
      // after this step, the nextState is populated
    }

    std::cout << "T rows: " << preOutputStates[0][0].rows() << " T cols: " << preOutputStates[0][0].columns() << std::endl;
    std::cout << "W4 rows: " << preOutputStates[0][1].rows() << " W4 cols: " << preOutputStates[0][1].columns() << std::endl;


    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
      beamSize = god.Get<size_t>("beam-size");
      }
    }
    Beams beams(batchSize);
    bool returnAlignment = god.Get<bool>("return-alignment");

    bestHyps_->CalcBeam(god, prevHyps, scorers_, filterIndices_, returnAlignment, beams, beamSizes);
    amunmt::CPU::BestHyps* bestHyps_cpu = static_cast<amunmt::CPU::BestHyps*>(bestHyps_.get()); //GRR ABSTRACTION

    //The same calculations are done in CalcBeam. What we do here is we identify what score corresponds to which
    //Beam/vocab item. We use that later to identify which states to dump.
    std::vector<std::pair<size_t, size_t> > beam_and_wordIDXs;
    for (size_t i = 0; i < bestHyps_cpu->bestKeys_.size(); i++) {
      size_t beamIDX = bestHyps_cpu->bestKeys_[i] / bestHyps_cpu->probs_cols;
      size_t wordIDX = bestHyps_cpu->bestKeys_[i] % bestHyps_cpu->probs_cols;
      beam_and_wordIDXs.push_back(std::pair<size_t, size_t>(beamIDX, wordIDX));
    }

    // Get the hypothesis so far
    std::vector<std::string> prevHypsStr;
    for (size_t h = 0; h < beams[0].size(); h++) {
      prevHypsStr.push_back(GetStringFromHypo(beams[0][h]));
    }

    //Drop states
    statesFile << GetStringsFromStates_presoftmax(preOutputStates[0], beam_and_wordIDXs, prevHypsStr);

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
          finishedFile << GetStringFromHypo(h) << " ||| " << h->GetCost()/(decoderStep+1) << '\n';
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

