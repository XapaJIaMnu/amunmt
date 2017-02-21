#pragma once
#include <memory>
#include "common/types.h"
#include "common/soft_alignment.h"
#include "common/hypothesis.h"
#include "common/scorer.h"

namespace amunmt {

class exploredItem;

class Hypothesis_states {
    public:
        exploredItem * parent = nullptr;
        HypothesisPtr cur_hypo;
        StatePtr cur_rnn_state;
        float accumulatedScore;

        Hypothesis_states(HypothesisPtr, StatePtr, float);

        std::string GetHypoWords(const God &god);
};

bool operator< (const Hypothesis_states &left, const Hypothesis_states &right) {
    return left.accumulatedScore < right.accumulatedScore;
}

bool operator> (const Hypothesis_states &left, const Hypothesis_states &right) {
    return left.accumulatedScore > right.accumulatedScore;
}

Hypothesis_states::Hypothesis_states(HypothesisPtr current, StatePtr currentState, float accumscore) :
    cur_hypo(current),
    cur_rnn_state(currentState),
    accumulatedScore(accumscore) {}

std::string Hypothesis_states::GetHypoWords(const God &god) {
  std::string out = "";
  HypothesisPtr hyp = cur_hypo;
  while (hyp != nullptr) {
    out = god.GetTargetVocab()[hyp->GetWord()] + " " + out;
    hyp = hyp->GetPrevHyp();
  }
  return out;
}

class exploredItem {
    public:
        float accumulatedScore;
        size_t word_idx; //Which word of the translation are we on
        std::vector<Hypothesis_states *> children;
        int child_idx = 0;

        Hypothesis_states * getNextChild();
        exploredItem(float, size_t, std::vector<Hypothesis_states *>);
        ~exploredItem();
};

exploredItem::exploredItem(float accumScore, size_t wordIDX, std::vector<Hypothesis_states *> my_children) : 
    accumulatedScore(accumScore),
    word_idx(wordIDX),
    children(my_children) {}

exploredItem::~exploredItem() {
    if (children.size() > 0) {
        for (auto item : children) {
            delete item; //Be a good Cronus
        }
        children.clear(); //Might run into double free here, not certain.
    }
}

Hypothesis_states * exploredItem::getNextChild() {
    if (child_idx == children.size() - 1) {
        //for (auto item : children) { //@TODO this creates issues when requesting next child because of hypothesis..
        //    delete item; //Be a good Cronus
        //} 
        //children.clear(); //Might run into double free here, not certain.
        return nullptr;
    } else {
        child_idx++;
        return children[child_idx];
    }
}

} //namespace
