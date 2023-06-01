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
        sgd,                            // stochastic gradient descent with microbatch support, good for small datasets, uses little memory, slow
        adam,                           // adaptive moment estimation, good for large datasets, uses more memory than sgd, fast, easy to tune
        sgdm,                           // stochastic gradient descent with momentum, good for large datasets, uses more memory than sgd, fast
        adam_with_decaying_momentum,    // adaptive moment estimation with decaying momentum, good for large datasets, uses more memory than adam, fast, difficult to tune
        sgdm_with_decaying_momentum     // stochastic gradient descent with momentum and decaying momentum, good for large datasets, uses more memory than sgdm, fast, difficult to tune
    };

    enum LossType {
        mse,                        // mean squared error, good for regression
        mae,                        // mean absolute error, good for regression
        smae,                       // scaled mean absolute error, good for regression
        categoricalCrossEntropy,    // good for classification
        binaryCrossEntropy,         // good for classification
    };

    enum LayerType {
        full,               // fully connected layer, good for classification and regression
        convolution2dValid, // convolutional layer, good for image classification
        concatenate,        // concatenates two layers
        flatten,            // flattens a layer
        normalize,          // normalizes a layer
        dropout             // randomly drops out a percentage of neurons
    };

    // Added the word "Default" after tanh because
    // the compiler was picking up tanh as the function tanh().
    // This feels bad to me, but I need a quick fix, and I'll ponder
    // a better naming situation later.
    enum ActivationType {
        relu,           // 0 to infinity, good for classification
        tanhDefault,    // -1 to 1, good for regression
        sigmoid,        // 0 to 1, good for classification
        leaky,          // approximately 0 to 1, good to prevent dead neurons
        softmax,        // 0 to 1, good for classification
        sigmoidApprox,  // 0 to 1, faster than sigmoid
        tanhApprox,     // -1 to 1, faster than tanh
        linear          // no activation, good for regression
    };

    enum TrainingRetentionPolicy {
        best, // accurate
        last  // fast and uses less disk space
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
            case dropout:
                return "dropout";
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
        if (layerType == "dropout") {
            return dropout;
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
            case sgd:
                return "SGD";
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
        if (optimizerType == "SGD") {
            return sgd;
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