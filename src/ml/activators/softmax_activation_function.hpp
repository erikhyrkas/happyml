//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP

#include "../types/tensor_views/value_transform_with_double_tensor_view.hpp"
#include "../../types/tensor_views/exponential_tensor_view.hpp"
#include "../../types/tensor_views/scalar_subtract_tensor_view.hpp"


namespace happyml {
// result tensor elements sum to 1, representing the percentage of importance of each element in original tensor
// usually represents a probability between 0 and 1 of each element in a classifications of multiple possibilities
    class SoftmaxActivationFunction : public ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            double max_input = input->max();
            // (input-max(input)) / sum(input-max(input)
            std::shared_ptr<BaseTensor> input_minus_max = make_shared<ScalarSubtractTensorView>(input, max_input);
            shared_ptr<BaseTensor> numerator = make_shared<ExponentialTensorView>(input_minus_max);
            double denominator = numerator->sum();
            shared_ptr<BaseTensor> output = make_shared<ScalarDivideTensorView>(numerator, denominator);
            return output;
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // shortcut. We don't explicitly compute the Jacobian matrix because this is always used before categorical cross-entropy loss
            return input;
        }
    };
}
#endif //HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
