#include "common/search.h"

#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

#include "common/hypostate.h"
#include <queue>

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

  std::vector<exploredItem *> parents;
  parents.reserve(400); //Say we explore 400 states;

  auto cmp =  [](Hypothesis_states * left, Hypothesis_states * right) 
    -> bool {return *left < *right;}; //We want a reverse priority here so that the smallest element is the first
  typedef std::priority_queue<Hypothesis_states*, std::vector<Hypothesis_states*>, decltype(cmp)> FineQ;
  std::vector<FineQ> all_queues;

  // We have a vector of priority queues 'all_queues' where all_queues[i] correspond to the priority queue with i words translated.
  // The vector 'coarse_q' holds indices to all_queues in a sorted PQ order, sorted using vecComp, and the approximated rest scores        .
  // vecComp: takes indices (word positions) left and right and finds which priority queue has higher top score.
  auto vecComp = [&all_queues](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right) -> bool
    { float leftscore = -999;
      float rightscore = -999;
      if (!all_queues[left.first].empty()) {
        leftscore = all_queues[left.first].top()->accumulatedScore + left.second;
      }
      if (!all_queues[right.first].empty()) {
        rightscore = all_queues[right.first].top()->accumulatedScore + right.second;
      }
      return leftscore > rightscore;};

  // We also want a normal vector sort in the order of the queues so that we can easily update it
  auto vecQueueSort = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right) -> bool
    {return left.first > right.first;};

  // index in all_queues; best score for the remaining part of the sentence
  std::vector<std::pair<size_t, float>> coarse_q;

  std::vector<float> future_scores; //This doesn't include the zeroth hypothesis (which is always the start of sentence)

  float topScore = 0;
  HypothesisPtr topHypo;

  //GREEDY BEST FIRST
  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Decode(god, state, nextState, beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = god.Get<size_t>("beam-size");
      }
    }
    Beams beams(batchSize);
    bool returnAlignment = god.Get<bool>("return-alignment");

    //@TODO VARY BEAM SIZE HERE
    bestHyps_->CalcBeam(god, prevHyps, scorers_, filterIndices_, returnAlignment, beams, beamSizes);
    std::sort(beams[0].begin(), beams[0].end(), [](HypothesisPtr& a, HypothesisPtr& b) -> bool { return a->GetCost() > b->GetCost(); });

    // Populate the future_scores with the score of the greedy
    //@TODO batching and multiple ensamblers (however it's spelled)
    if (decoderStep == 0) {
      future_scores.push_back(beams[0][0]->GetCost());
    } else {
      future_scores.push_back(beams[0][0]->GetCost() - beams[0][0]->GetPrevHyp()->GetCost());
    }
    //LOG(info) << "BLA " << future_scores[decoderStep];


    //CreateChildren
    std::vector<Hypothesis_states *> initial_children;
    initial_children.reserve(beams[0].size());

    for (HypothesisPtr hypo : beams[0]) {
      StatePtr curState;
      curState.reset(scorers_[0]->NewState()); //@TODO ensembling
      std::vector<HypothesisPtr> survivors = {hypo};
      scorers_[0]->AssembleBeamState(*nextStates[0], survivors, *curState);

      Hypothesis_states * hypostate = new Hypothesis_states(hypo, curState, hypo->GetCost()); //This is a partial UNNORMALIZED SCORE
      initial_children.push_back(hypostate);
    }

    //Create parent:
    //@TODO the parent shouldn't include the first item because it's used now?
    exploredItem * expl = new exploredItem(prevHyps[0]->GetCost(), decoderStep, initial_children);
    parents.push_back(expl);
    //Point each child to its parent:
    for (Hypothesis_states * hypostate : initial_children) {
      hypostate->parent = expl;
    }

    all_queues.push_back(FineQ(cmp, {expl->getNextChild()}));

    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        histories->at(i)->Add(beams[i], histories->at(i)->size() == 3 * sentences.at(i)->GetWords().size());
      }
    }

    Beam survivors = std::vector<HypothesisPtr>{initial_children[0]->cur_hypo};
    /*
    for (size_t batchID = 0; batchID < batchSize; ++batchID) {
      for (auto& h : beams[batchID]) {
        if (h->GetWord() != EOS) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchID];
        }
      }
    }

    if (survivors.size() == 0) {
      break;
    }
    */
    if (survivors[0]->GetWord() == EOS) {
      topScore = survivors[0]->GetCost()/(decoderStep + 1);
      topHypo = survivors[0];
      break;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);

  }

  //Our 0th index is actually the first word translated, because the 0th is always the <s> token
  float accumulatedFutureScoreApproximation = 0;
  coarse_q.resize(all_queues.size());
  for (int i = coarse_q.size() - 1; i>=0; i--) {
    accumulatedFutureScoreApproximation += future_scores[i];
    coarse_q[i] = (std::pair<size_t, float>(i, accumulatedFutureScoreApproximation));
  }
  std::sort(coarse_q.begin(), coarse_q.end(), vecComp);

/*
  for (auto& q : all_queues) {
    auto& e = q.top();
    LOG(info) << q.size();
    if (e->parent) {
      LOG(info) << e->accumulatedScore << " " << e->parent->word_idx;
    } else {
      LOG(info) << e->accumulatedScore << " NO";
    }
    LOG(info) << "---";
  }
*/
  //Now do the refinements;
  int limit = 0;
  int completions = 1;
  int refinements = 0;
  while (limit < 5000) {
    limit++;

    //Get the top element we want to expand. We might want to settle for second best if a queue is empty
    Hypothesis_states * top = nullptr;
    size_t chosen_q = 0;
    size_t coarse_q_idx = 0;
    for (std::pair<size_t, float> sorted_element : coarse_q) {
      if (!all_queues[sorted_element.first].empty()) {
        top = all_queues[sorted_element.first].top();
        all_queues[sorted_element.first].pop();
        chosen_q = sorted_element.first;
        break;
      }
      coarse_q_idx++;
    }
    //We couldn't find a top, break
    if (!top) {
      break;
    }

    if (limit % 100 == 0) {
      /*LOG(info) << "On expansion: " << limit;
      LOG(info) << "Chosen Q: " << chosen_q << " out of: " << coarse_q.size();
      LOG(info) << "->size Q: " << all_queues[chosen_q].size();
      LOG(info) << "score: Q: " << top->accumulatedScore + coarse_q[coarse_q_idx].second;

      for (int i = 0; i<coarse_q.size(); i++) {
        if (!all_queues[coarse_q[i].first].empty()) {
          std::cout << "Q " << coarse_q[i].first << " score: " << coarse_q[i].second + all_queues[coarse_q[i].first].top()->accumulatedScore << " ";
        }
        std::cout << std::endl;
      }*/
    }

    //Add next sibling to the priority_queue
    //Push next child if such exists:
    Hypothesis_states * next_child = top->parent->getNextChild();
    if (next_child) {
      all_queues[chosen_q].push(next_child);
    }
    
    //If we reach an end of sentence we don't want to expand
    if (top->cur_hypo->GetWord() == EOS) {
      completions++;
      //Check if it's better than the current one
      float normalisedScore = top->accumulatedScore/(top->parent->word_idx + 1);
      LOG(info) << "New completion found: Raw score: " << top->accumulatedScore << " num words: " << top->parent->word_idx + 1;
      LOG(info) << "New completion found: score: " << normalisedScore << " Previous best: " << topScore;
      LOG(info) << "--> " << top->GetHypoWords(god);
      if (normalisedScore > topScore) {
        topScore = normalisedScore;
        topHypo = top->cur_hypo;
        refinements++;
      }
      continue;
    }

    //Expand once. Assume we only have one decoder and no batches, because we are cheap. Sue us
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];

      State &nextState = *nextStates[i];

      scorer.Decode(god, *top->cur_rnn_state.get(), nextState, beamSizes);
    }

    Beams beams(batchSize);
    bool returnAlignment = god.Get<bool>("return-alignment");

    bestHyps_->CalcBeam(god, {top->cur_hypo}, scorers_, filterIndices_, returnAlignment, beams, beamSizes);
    std::sort(beams[0].begin(), beams[0].end(), [](HypothesisPtr& a, HypothesisPtr& b) -> bool { return a->GetCost() > b->GetCost(); });

    //CreateChildren
    std::vector<Hypothesis_states *> children;
    children.reserve(beams[0].size());

    for (HypothesisPtr hypo : beams[0]) {
      StatePtr curState;
      curState.reset(scorers_[0]->NewState()); //@TODO ensembling
      std::vector<HypothesisPtr> survivors = {hypo};
      scorers_[0]->AssembleBeamState(*nextStates[0], survivors, *curState);

      Hypothesis_states * hypostate = new Hypothesis_states(hypo, curState, hypo->GetCost()); //This is a partial UNNORMALIZED SCORE
      children.push_back(hypostate);
    }

    //Create parent:
    //@TODO the parent shouldn't include the first item because it's used now?
    exploredItem * expl = new exploredItem(prevHyps[0]->GetCost(), chosen_q + 1, children);
    parents.push_back(expl);
    //Point each child to its parent:
    for (Hypothesis_states * hypostate : children) {
      hypostate->parent = expl;
    }

    //Put in the appropriate queue
    if (all_queues.size() <= chosen_q + 1) {
      all_queues.push_back(FineQ(cmp, {expl->getNextChild()}));
      //Put a placeholder in coarse Q which will be updated later
      coarse_q.push_back(std::pair<size_t, float>(chosen_q + 1, 0));
    } else {
      all_queues[chosen_q + 1].push(expl->getNextChild());
    }
    
    //Update predictions

    //Update future score if necessary
    //@TODO batching and multiple ensamblers (however it's spelled)
    bool futurescoreUpdated = false;
    if (future_scores.size() <= chosen_q + 1) {
      //If we have expanded the hypothesis past the previous maximum we push an updates future score
      future_scores.push_back(beams[0][0]->GetCost() - beams[0][0]->GetPrevHyp()->GetCost());
      futurescoreUpdated = true;
    } else if (future_scores[chosen_q + 1] < (beams[0][0]->GetCost() - beams[0][0]->GetPrevHyp()->GetCost())){
      //LOG(info) << "Future score update: Before: " << future_scores[chosen_q + 1];
      future_scores[chosen_q + 1] = (beams[0][0]->GetCost() - beams[0][0]->GetPrevHyp()->GetCost());
      //Updated future score:
      //LOG(info) << "Future score update: After: " << future_scores[chosen_q + 1];
      futurescoreUpdated = true;
    }

    //Update the estimates: @TODO do less work here.
    if (futurescoreUpdated) {
      std::sort(coarse_q.begin(), coarse_q.end(), vecQueueSort); //Restore the queue in default order:
      accumulatedFutureScoreApproximation = 0;
      for (int i = coarse_q.size() - 1; i>=0; i--) {
        accumulatedFutureScoreApproximation += future_scores[i];
        coarse_q[i] = (std::pair<size_t, float>(i, accumulatedFutureScoreApproximation));
      }
      std::sort(coarse_q.begin(), coarse_q.end(), vecComp);
    } else {
      //Resort/update the queues
      std::sort(coarse_q.begin(), coarse_q.end(), vecComp); //Just attempt to heapify again
    }

  }
  LOG(info) << "Top scoring hypothesis' score: " << topScore;
  LOG(info) << "Completions: " << completions;
  LOG(info) << "Refinements: " << refinements;

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

