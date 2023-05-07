//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP

#include "../types/tensor_views/tensor_value_transform_3_view.hpp"
#include "../types/tensor_views/tensor_diagonal_view.hpp"
#include "../types/tensor_views/tensor_value_transform_4_view.hpp"

namespace happyml {
// result tensor elements sum to 1, representing the percentage of importance of each element in original tensor
// usually represents a probability between 0 and 1 of each element in a classifications of multiple possibilities
    class SoftmaxActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            double normalization_constant = 0.0;
            size_t records = input->size();
            for (size_t next_record = 0; next_record < records; next_record++) {
                normalization_constant += std::exp(input->getValue(next_record));
            }
            auto transformFunction = [](float original, double constant) {
                return (float) (std::exp(original) / constant);
            };
            return std::make_shared<happyml::TensorValueTransform3View>(input, transformFunction, normalization_constant);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            return activate(input);
        }
    };
}
#endif //HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
