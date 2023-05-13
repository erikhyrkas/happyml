//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_LEAKY_RELU_ACTIVATION_FUNCTION_HPP
#define HAPPYML_LEAKY_RELU_ACTIVATION_FUNCTION_HPP


namespace happyml {
// small negative number to infinity
    class LeakyReLUActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // avoid branching in a loop. give negative values a small value.
                return ((float) (original < 0.0f)) * (0.01f * original) + ((float) (original >= 0.0f)) * original;
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // avoid branching in a loop. give negative values a small value.
                return ((float) (original < 0.0f)) * 0.01f + ((float) (original >= 0.0f)) * 1.0f;
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }
    };
}

#endif //HAPPYML_LEAKY_RELU_ACTIVATION_FUNCTION_HPP
