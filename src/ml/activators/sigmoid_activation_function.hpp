//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SIGMOID_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SIGMOID_ACTIVATION_FUNCTION_HPP

namespace happyml {
// I found this article useful in verifying the formula: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
// I also checked this understanding here: https://medium.com/@DannyDenenberg/derivative-of-the-sigmoid-function-774446dfa462
// 0 to 1
    class SigmoidActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                return 1.0f / (1.0f + exp(-1.0f * original));
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // result = sigmoid(x) * (1.0 - sigmoid(x))
            auto transformFunction = [](float original) {
                auto sig = 1.0f / (1.0f + exp(-1.0f * original));
                return sig * (1.f - sig);
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }
    };
}
#endif //HAPPYML_SIGMOID_ACTIVATION_FUNCTION_HPP
