//
// Created by Erik Hyrkas on 10/25/2022.
//

#ifndef MICROML_ACTIVATION_HPP
#define MICROML_ACTIVATION_HPP

#include <iostream>
#include "quarter_float.hpp"
#include "tensor.hpp"

namespace microml {

    class ActivationFunction {
    public:
        virtual std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) = 0;

        virtual std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) = 0;
    };

    class LinearActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            // copy input to output without changing it
            return std::make_unique<TensorNoOpView>(input);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // sent all 1s to output in the same shape as input
            return std::make_shared<UniformTensor>(input->row_count(), input->column_count(), input->channel_count(), 1.0f);
        }

    };

    class LeakyReLUActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // avoid branching in a loop. give negative values a small value.
                return ((float) (original < 0.0f)) * (0.01f * original) + ((float) (original >= 0.0f)) * original;
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // avoid branching in a loop. give negative values a small value.
                return ((float) (original < 0.0f)) * 0.01f + ((float) (original >= 0.0f)) * 1.0f;
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }
    };

    class ReLUActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                return std::max(original, 0.0f);
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                return original > 0.0f; // the greater than operator returns 1 or 0
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }
    };

    class SoftmaxActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            float largest_value = input->max();
            double sum = 0.0;
            if (input->row_count() == 1 && input->column_count() > 0) {
                for (size_t col = 0; col < input->column_count(); col++) {
                    sum += std::exp(input->get_val(0, col, 0) - largest_value);
                }
            } else if (input->column_count() == 1 && input->row_count() > 0) {
                for (size_t row = 0; row < input->row_count(); row++) {
                    sum += std::exp(input->get_val(row, 0, 0) - largest_value);
                }
            } else {
                throw std::exception("Softmax supports input with a single row or single column.");
            }
            std::vector<double> constants{largest_value, sum};
            auto transformFunction = [](float original, std::vector<double> constants) {
                return ((double) std::expf(original - (float) constants[0])) / constants[1];
            };
            return std::make_shared<TensorValueTransform2View>(input, transformFunction, constants);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            std::shared_ptr<BaseTensor> softmax_out = activate(input);
            auto negative = std::make_shared<TensorMultiplyByScalarView>(softmax_out, -1.0f);
            auto reshape = std::make_shared<TensorReshapeView>(softmax_out, softmax_out->column_count(),
                                                               softmax_out->row_count());
            auto dot_product_view = std::make_shared<TensorDotTensorView>(negative, reshape);
            auto diag = std::make_shared<TensorDiagonalView>(softmax_out);
            return std::make_shared<TensorAddTensorView>(dot_product_view, diag);
        }
    };


// There may be faster means of approximating this. See: https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
// If I go this route, I'd probably make a whole new class and let the caller decide on whether to approximate or not
// maybe "class SigmoidApproximationActivationFunction"
    class SigmoidActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            // Is this needed? I don't see why.
//        if (input->row_count() + input->column_count() != 1) {
//            throw std::exception("Sigmoid supports input with a single row or single column.");
//        }
            auto transformFunction = [](float original) {
                return 1.0f / (1.0f + std::expf(-1.0f * original));
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            // result = sigmoid(x) * (1.0 - sigmoid(x))
            // sigmoid(x) =
            std::shared_ptr<BaseTensor> sigmoid = activate(input);
            // 1.0 - sigmoid == -sigmoid + 1 =
            auto negative_sigmoid = std::make_shared<TensorMultiplyByScalarView>(input, -1.0);
            auto plus_one = std::make_shared<TensorAddScalarView>(negative_sigmoid, 1.0);
            return std::make_shared<TensorDotTensorView>(sigmoid, plus_one);
        }
    };

    class TanhActivationFunction : public ActivationFunction {
        std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) override {
            // Is this needed? I don't see why.
//        if (input->row_count() + input->column_count() != 1) {
//            throw std::exception("Sigmoid supports input with a single row or single column.");
//        }
            auto transformFunction = [](float original) {
                // optimization or waste of energy?
                // tanh(x) = 2 * sigmoid(2x) - 1
//            const float two_x = (2 * original);
//            const float sigmoid = 1.0f / (1.0f + std::expf(-1.0f * two_x));
//            return (2 * sigmoid) - 1;
                return tanh(original);
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }

        std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) override {
            auto transformFunction = [](float original) {
                // 1 - tanh^2{x}
                const float th = tanh(original);
                return 1 - (th * th);
            };
            return std::make_shared<TensorValueTransformView>(input, transformFunction);
        }
    };
}
#endif //MICROML_ACTIVATION_HPP
