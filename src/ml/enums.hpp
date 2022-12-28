//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//
#ifndef HAPPYML_ENUMS_HPP
#define HAPPYML_ENUMS_HPP

#include <iostream>

using namespace std;

namespace happyml {

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
        tanhApprox
    };

    enum TrainingRetentionPolicy {
        best, // accurate
        last  // fast
    };

    string activationTypeToString(ActivationType activationType) {
        switch (activationType) {
            case relu:
                return "relu";
            case tanhDefault:
                return "tanh";
            case sigmoid:
                return "sigmoid";
            case leaky:
                return "leaky";
            case softmax:
                return "softmax";
            case sigmoidApprox:
                return "sigmoidApprox";
            case tanhApprox:
                return "tanhApprox";
        }
        throw exception("Unknown Activation Type");
    }

    ActivationType stringToActivationType(const string &activationType) {
        if (activationType == "relu") {
            return relu;
        }
        if (activationType == "tanh") {
            return tanhDefault;
        }
        if (activationType == "sigmoid") {
            return sigmoid;
        }
        if (activationType == "leaky") {
            return leaky;
        }
        if (activationType == "softmax") {
            return softmax;
        }
        if (activationType == "sigmoidApprox") {
            return sigmoidApprox;
        }
        if (activationType == "tanhApprox") {
            return tanhApprox;
        }
        throw exception("Unknown Activation Type");
    }

    string nodeTypeToString(NodeType nodeType) {
        switch (nodeType) {
            case full:
                return "full";
            case convolution2dValid:
                return "convolution2dValid";
        }
        throw exception("Unknown Node Type");
    }

    NodeType stringToNodeType(const string &nodeType) {
        if (nodeType == "full") {
            return full;
        }
        if (nodeType == "convolution2dValid") {
            return convolution2dValid;
        }
        throw exception("Unknown Node Type");
    }

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

#endif // HAPPYML_ENUMS_HPP