//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP

#include "../types/tensor_views/value_transform_with_double_tensor_view.hpp"


namespace happyml {
// result tensor elements sum to 1, representing the percentage of importance of each element in original tensor
// usually represents a probability between 0 and 1 of each element in a classifications of multiple possibilities
    class SoftmaxActivationFunction : public ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            double max_input = input->max();
            double normalization_constant = 0.0;
            size_t records = input->size();
            for (size_t next_record = 0; next_record < records; next_record++) {
                double exp_val = std::exp(input->getValue(next_record) - max_input);
#ifdef DEBUG_TRAIN_NAN
                if (isnan(exp_val)) {
                    input->print();
                    throw std::runtime_error("SoftmaxActivationFunction::activate: exp_val is NaN");
                }
#endif
                normalization_constant += exp_val;
            }
            if (normalization_constant == 0) {
                normalization_constant = 1e-8;
            } else if (std::isnan(normalization_constant)) {
                normalization_constant = 1;
            }
            auto transformFunction = [max_input](float original, double constant) {
                return (float) (std::exp(original - max_input) / constant);
            };
            auto result = std::make_shared<ValueTransformWithDoubleTensorView>(input, transformFunction, normalization_constant);
            return result;
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // shortcut
            return input;
        }
    };
}
#endif //HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
