//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SIGMOID_APPROX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SIGMOID_APPROX_ACTIVATION_FUNCTION_HPP

namespace happyml {
// There may be faster means of approximating sigmoid. See: https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
// f(x) = 0.5 * (x / (1 + abs(x)) + 1)
// 0 to 1
    class SigmoidApproximationActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                return 0.5f * ((original / (1.0f + abs(original))) + 1);
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // result = sigmoid(x) * (1.0 - sigmoid(x))
            auto transformFunction = [](float original) {
                // todo: validate math.
                auto sig = 0.5f * ((original / (1.0f + abs(original))) + 1);
                return sig * (1.f - sig);
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }
    };
}
#endif //HAPPYML_SIGMOID_APPROX_ACTIVATION_FUNCTION_HPP
