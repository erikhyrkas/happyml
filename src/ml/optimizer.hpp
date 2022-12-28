//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_OPTIMIZER_HPP
#define HAPPYML_OPTIMIZER_HPP

#include "neural_network_function.hpp"

// Optimizers are the strategy applied to find the optimal results
// The optimizer takes in:
// 1. The results that the model predicted
// 2. The results that the model should have gotten (the "truth")
// The optimizer uses a loss function (which is simply a bit of math to calculate how close a prediction is
// to the true answer) to compare the two and then updates the weights.

// Implementation note:
// After a lot of futzing around, I decided to build the optimizer into the neural network nodes directly.
// I treat the optimizer as a factory that generates the needed learning functions.
// To me, this made the resulting code seem more logical and didn't require weird or difficult to understand
// code.
// You'll notice that not all neural network functions are optimizer specific. Technically, you only need an
// optimizer to train a model. You don't need one to make predictions. Because optimizers save state while
// making a prediction to be able to later learn, this can be wasteful if you are never going to use that extra
// state.

namespace happyml {

    class Optimizer {
    public:
        virtual shared_ptr<NeuralNetworkFunction> createConvolutional2d(const string &label, vector<size_t> input_shape,
                                                                        size_t filters, size_t kernel_size,
                                                                        uint8_t bits) = 0;

        virtual shared_ptr<NeuralNetworkFunction> createFullyConnectedNeurons(const string &label, size_t input_size, size_t output_size, uint8_t bits) = 0;

        virtual shared_ptr<NeuralNetworkFunction> createBias(const string &label, vector<size_t> input_shape, vector<size_t> output_shape, uint8_t bits) = 0;
    };
}
#endif //HAPPYML_OPTIMIZER_HPP
