//
// Created by Erik Hyrkas on 12/26/2022.
//
#ifndef MICROML_ENUMS_HPP
#define MICROML_ENUMS_HPP

#include <iostream>

using namespace std;

namespace microml {

    enum OptimizerType { microbatch, adam };

    enum LossType { mse };

    enum NodeType { full, convolution2dValid };

    // Added the word "Default" after tanh because
    // the compiler was picking up tanh as the function tanh().
    // This feels bad to me, but I need a quick fix, and I'll ponder
    // a better naming situation later.
    enum ActivationType {
        relu,
        tanhDefault,
        sigmoid,
        leaky,
        softmax,
        sigmoidApprox,
        tanhApprox };

    enum TrainingRetentionPolicy {
        best, // accurate
        last  // fast
    };

    string lossTypeToString(LossType lossType) {
        switch (lossType) {
            case mse:
                return "mse";
        }
        throw exception("Unknown Loss Type");
    }

    LossType stringToLossType(const string &lossType) {
        if (lossType == "mse") {
            return mse;
        }
        throw exception("Unknown Loss Type");
    }

    string optimizerTypeToString(OptimizerType optimizerType) {
        switch (optimizerType) {
            case microbatch:
                return "Micro Batch";
            case adam:
                return "Adam";
        }
        throw exception("Unknown Optimizer Type");
    }

    OptimizerType stringToOptimizerType(const string &optimizerType) {
        if (optimizerType == "Micro Batch") {
            return microbatch;
        }
        if (optimizerType == "Adam") {
            return adam;
        }
        throw exception("Unknown Optimizer Type");
    }
}

#endif // MICROML_ENUMS_HPP