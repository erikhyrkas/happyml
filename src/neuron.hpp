//
// Created by Erik Hyrkas on 11/20/2022.
//

#ifndef MICROML_NEURON_HPP
#define MICROML_NEURON_HPP

#include "tensor.hpp"
#include "activation.hpp"

// A single neron is enough to do linear regression - just use LinearActivationFunction
// It has weights for inputs, uses an activation function, and then adds a bias.
class Neuron {
public:
    Neuron(std::shared_ptr<ActivationFunction> activationFunction, size_t input_rows, size_t input_cols, size_t input_channels, int initial_quarter_bias, uint32_t seed) {
        this->bias = 0;
        this->weights = std::make_shared<TensorFromRandom>(input_rows, input_cols, input_channels, initial_quarter_bias, seed);
        this->activationFunction = activationFunction;
    }
    Neuron(std::shared_ptr<ActivationFunction> activationFunction, size_t input_rows, size_t input_cols, size_t input_channels) {
        this->bias = 0;
        this->weights = std::make_shared<TensorFromRandom>(input_rows, input_cols, input_channels, 8);
        this->activationFunction = activationFunction;
    }

    float activate(std::shared_ptr<BaseTensor> input) {
        auto weighted_inputs = std::make_shared<TensorDotTensorView>(input, weights);
        auto biased_weighted_inputs = std::make_shared<TensorAddScalarView>(weighted_inputs, bias);
        activationFunction->activate(biased_weighted_inputs);
    }
private:
    std::shared_ptr<ActivationFunction> activationFunction;
    std::shared_ptr<BaseTensor> weights;
    float bias;
};
#endif //MICROML_NEURON_HPP
