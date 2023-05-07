//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_LINEAR_ACTIVATION_FUNCTION_HPP
#define HAPPYML_LINEAR_ACTIVATION_FUNCTION_HPP

#include <iostream>

namespace happyml {
// also known as the "identity" activation function.
// do nothing. useful for basic linear regression where we don't have an activation function.
    class LinearActivationFunction : public ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            // copy input to output without changing it
            return input;
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // sent all 1s to output in the same shape as input
            return std::make_shared<UniformTensor>(input->getShape(), 1.0f);
        }
    };
}
#endif //HAPPYML_LINEAR_ACTIVATION_FUNCTION_HPP
