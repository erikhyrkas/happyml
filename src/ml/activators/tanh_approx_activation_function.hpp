//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TANH_APPROX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_TANH_APPROX_ACTIVATION_FUNCTION_HPP

#include "../activation.hpp"
#include "../../types/base_tensors.hpp"
#include "../../util/basic_profiler.hpp"

namespace happyml {
// approximate tanhActivation
// I read about this here: https://www.ipol.im/pub/art/2015/137/article_lr.pdf
    class TanhApproximationActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            PROFILE_BLOCK(profileBlock);
            auto transformFunction = [](float original) {
                // tanhActivation(x) = 2 * sigmoid(2x) - 1
                const float twoX = (2 * original);
                auto sigmoid = 1.0f / (1.0f + exp(-1.0f * twoX));
//                auto sigmoid = 0.5f * ((original / (1.0f + std::abs(original))) + 1); //super approx
                return (2 * sigmoid) - 1;
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            PROFILE_BLOCK(profileBlock);
            // result = sigmoid(x) * (1.0 - sigmoid(x))
            auto transformFunction = [](float original) {
                // todo: validate math.
                // 1 - tanhActivation^2{x}
                const float two_x = (2 * original);
                auto sigmoid = 1.0f / (1.0f + exp(-1.0f * two_x));
//                auto sigmoid = 0.5f * ((original / (1.0f + std::abs(original))) + 1); //super approx
                const float th = (2 * sigmoid) - 1;
                return 1 - (th * th);
            };
            return std::make_shared<happyml::ValueTransformTensorView>(input, transformFunction);
        }
    };
}
#endif //HAPPYML_TANH_APPROX_ACTIVATION_FUNCTION_HPP
