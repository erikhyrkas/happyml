//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_RELU_ACTIVATION_FUNCTION_HPP
#define HAPPYML_RELU_ACTIVATION_FUNCTION_HPP

#include <iostream>

namespace happyml {
// Useful in the hidden layers of a neural network, especially deep neural networks and convolutional neural networks.
// 0 to infinity
    class ReLUActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                return std::max(original, 0.0f);
            };
            return std::make_shared<happyml::TensorValueTransformView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // derivative original == 0 is undefined.
                if (original > 0.f) {
                    return 1.0f;
                }
                return 0.f;
            };
            return std::make_shared<happyml::TensorValueTransformView>(input, transformFunction);
        }
    };
}
#endif //HAPPYML_RELU_ACTIVATION_FUNCTION_HPP
