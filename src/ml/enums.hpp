//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_ENUMS_HPP
#define HAPPYML_ENUMS_HPP

#include <iostream>

using namespace std;

namespace happyml {

    enum OptimizerType {
        microbatch, adam_with_decaying_momentum, sgdm, adam, sgdm_with_decaying_momentum
    };

    enum LossType {
        mse,
        mae,
        smae,
        categoricalCrossEntropy,
        binaryCrossEntropy,
    };

    enum LayerType {
        full, convolution2dValid, concatenate, flatten, normalize
    };

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
        tanhApprox,
        linear
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
            case linear:
                return "linear";
        }
        throw runtime_error("Unknown Activation Type");
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
        if (activationType == "linear") {
            return linear;
        }
        throw runtime_error("Unknown Activation Type");
    }

    string nodeTypeToString(LayerType layerType) {
        switch (layerType) {
            case full:
                return "full";
            case convolution2dValid:
                return "convolution2dValid";
            case concatenate:
                return "concatenate";
            case flatten:
                return "flatten";
            case normalize:
                return "normalize";
        }
        throw runtime_error("Unknown Node Type");
    }

    LayerType stringToNodeType(const string &layerType) {
        if (layerType == "full") {
            return full;
        }
        if (layerType == "convolution2dValid") {
            return convolution2dValid;
        }
        if (layerType == "concatenate") {
            return concatenate;
        }
        if (layerType == "flatten") {
            return flatten;
        }
        if (layerType == "normalize") {
            return normalize;
        }
        string error = "Unknown Node Type: " + layerType;
        throw runtime_error(error);
    }

    string lossTypeToString(LossType lossType) {

        switch (lossType) {
            case mse:
                return "mse";
            case mae:
                return "mae";
            case smae:
                return "smae";
            case categoricalCrossEntropy:
                return "categoricalCrossEntropy";
            case binaryCrossEntropy:
                return "binaryCrossEntropy";
        }

        throw runtime_error("Unknown Loss Type");
    }

    LossType stringToLossType(const string &lossType) {
        if (lossType == "mse") {
            return mse;
        }
        if (lossType == "mae") {
            return mae;
        }
        if (lossType == "smae") {
            return smae;
        }
        if (lossType == "categoricalCrossEntropy") {
            return categoricalCrossEntropy;
        }
        if (lossType == "binaryCrossEntropy") {
            return binaryCrossEntropy;
        }
        throw runtime_error("Unknown Loss Type");
    }

    string optimizerTypeToString(OptimizerType optimizerType) {
        switch (optimizerType) {
            case microbatch:
                return "Micro Batch";
            case adam_with_decaying_momentum:
                return "Adam with Decaying Momentum";
            case adam:
                return "Adam";
            case sgdm_with_decaying_momentum:
                return "SGDM with Decaying Momentum";
            case sgdm:
                return "SGDM";
        }
        throw runtime_error("Unknown Optimizer Type");
    }

    OptimizerType stringToOptimizerType(const string &optimizerType) {
        if (optimizerType == "Micro Batch") {
            return microbatch;
        }
        if (optimizerType == "Adam with Decaying Momentum") {
            return adam_with_decaying_momentum;
        }
        if (optimizerType == "Adam") {
            return adam;
        }
        if (optimizerType == "SGDM with Decaying Momentum") {
            return sgdm_with_decaying_momentum;
        }
        if (optimizerType == "SGDM") {
            return sgdm;
        }
        throw runtime_error("Unknown Optimizer Type");
    }
}

#endif // HAPPYML_ENUMS_HPP