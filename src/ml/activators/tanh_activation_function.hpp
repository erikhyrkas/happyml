//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TANH_ACTIVATION_FUNCTION_HPP
#define HAPPYML_TANH_ACTIVATION_FUNCTION_HPP

#include "../activation.hpp"

namespace happyml {
// Generally used for classification
// -1 to 1
    class TanhActivationFunction : public ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // optimization or waste of energy?
                // tanhActivation(x) = 2 * sigmoid(2x) - 1
//            const float two_x = (2 * original);
//            const float sigmoid = 1.0f / (1.0f + std::expf(-1.0f * two_x));
//            return (2 * sigmoid) - 1;
                return tanh(original);
            };
            return std::make_shared<ValueTransformTensorView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // 1 - tanhActivation^2{x}
                const float th = tanh(original);
                return 1 - (th * th);
            };
            return std::make_shared<ValueTransformTensorView>(input, transformFunction);
        }
    };
}
#endif //HAPPYML_TANH_ACTIVATION_FUNCTION_HPP
