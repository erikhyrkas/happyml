//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
#define HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP

#include "../types/tensor_views/tensor_value_transform_2_view.hpp"
#include "../types/tensor_views/tensor_diagonal_view.hpp"

namespace happyml {
// result tensor elements sum to 1, representing the percentage of importance of each element in original tensor
// usually represents a probability between 0 and 1 of each element in a classifications of multiple possibilities
    class SoftmaxActivationFunction : public happyml::ActivationFunction {
    public:
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            float largestValue = input->max();
            double sum = 0.0;
            if (input->rowCount() == 1 && input->columnCount() > 0) {
                for (size_t col = 0; col < input->columnCount(); col++) {
                    sum += exp(input->getValue(0, col, 0) - largestValue);
                }
            } else if (input->columnCount() == 1 && input->rowCount() > 0) {
                for (size_t row = 0; row < input->rowCount(); row++) {
                    sum += exp(input->getValue(row, 0, 0) - largestValue);
                }
            } else {
                throw std::exception("Softmax supports input with a single row or single column.");
            }
            std::vector<double> constants{largestValue, sum};
            auto transformFunction = [](float original, std::vector<double> constants) {
                return (float) (((double) expf(original - (float) constants[0])) / constants[1]);
            };
            return std::make_shared<happyml::TensorValueTransform2View>(input, transformFunction, constants);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // fixme: broken. producing the wrong shape output. (The output shape is too small.)
            std::shared_ptr<BaseTensor> softmaxOut = activate(input);
            auto negative = std::make_shared<TensorMultiplyByScalarView>(softmaxOut, -1.0f);
            auto reshape = std::make_shared<happyml::TensorReshapeView>(softmaxOut, softmaxOut->columnCount(),
                                                                        softmaxOut->rowCount());
            auto dot_product_view = std::make_shared<happyml::TensorMatrixMultiplyTensorView>(negative, reshape);
            auto diag = std::make_shared<happyml::TensorDiagonalView>(softmaxOut);
            std::cout << "softmax: work in progress... fix me." << std::endl;
            softmaxOut->print();
            diag->print();
            dot_product_view->print();
            return std::make_shared<TensorAddTensorView>(dot_product_view, diag);
        }
    };
}
#endif //HAPPYML_SOFTMAX_ACTIVATION_FUNCTION_HPP
