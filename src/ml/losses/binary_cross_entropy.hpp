//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_BINARY_CROSS_ENTROPY_HPP
#define HAPPYML_BINARY_CROSS_ENTROPY_HPP

#include "../../types/tensor_views/tensor_minus_scalar_view.hpp"
#include "../../types/tensor_views/tensor_matrix_divide_tensor_view.hpp"

namespace happyml {
    // The BinaryCrossEntropyLossFunction class implements the binary cross-entropy loss function for
    // binary classification problems. It computes the loss and its derivative with respect to the
    // model's predictions.
    class BinaryCrossEntropyLossFunction : public LossFunction {
    public:
        // calculate_error_for_one_prediction computes the element-wise error for a single prediction
        // using the binary cross-entropy formula:
        // -truth * log(prediction) - (1 - truth) * log(1 - prediction)
        shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            auto epsilon = 1e-8f;
            auto clip_prediction = make_shared<TensorClipView>(prediction, epsilon, 1.0f - epsilon);
            auto clip_one_minus_prediction = make_shared<TensorClipView>(make_shared<TensorMinusScalarView>(1.0f, prediction), epsilon, 1.0f - epsilon);

            auto log_prediction = make_shared<TensorLogView>(clip_prediction);
            auto log_one_minus_prediction = make_shared<TensorLogView>(clip_one_minus_prediction);

            auto one_minus_truth = make_shared<TensorMinusScalarView>(1.0f, truth);

            auto truth_matmul_log_prediction = make_shared<TensorElementWiseMultiplyByTensorView>(truth, log_prediction);
            auto one_minus_truth_matmul_log_one_minus_prediction = make_shared<TensorElementWiseMultiplyByTensorView>(one_minus_truth, log_one_minus_prediction);

            auto total_error = make_shared<TensorAddTensorView>(truth_matmul_log_prediction, one_minus_truth_matmul_log_one_minus_prediction);
            auto negative_total_error = make_shared<TensorMultiplyByScalarView>(total_error, -1.0f);
            return negative_total_error;
        }

        // computeBatchLoss computes the average loss for a batch of predictions
        float computeBatchLoss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: -truth_i * log(prediction_i) - (1 - truth_i) * log(1 - prediction_i)
            // For a single prediction: binary cross-entropy = sum(total_error)
            // For a batch, we take the average error: avg(sum(total_error))
            auto batch_size = static_cast<float>(total_error->size());
            auto sum_error = total_error->sum();
            auto avg_error = sum_error / batch_size;
            return static_cast<float>(avg_error);
        }

        // calculate_batch_loss_derivative computes the derivative of the binary cross-entropy loss
// with respect to the predictions for a batch of examples
        shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               vector<shared_ptr<BaseTensor>> &truths,
                                                               vector<shared_ptr<BaseTensor>> &predictions) override {
            // batch_size should be equal to the size of truths or predictions
            auto batch_size = static_cast<float>(truths.size());

            shared_ptr<BaseTensor> accumulatedLossDerivative;
            for (size_t i = 0; i < truths.size(); i++) {
                // Derivative of binary cross-entropy = -(truth/prediction - (1 - truth) / (1 - prediction)) / batch_size

                // truth / prediction
                auto truth_div_prediction = make_shared<TensorElementWiseDivideByTensorView>(truths[i], predictions[i]);

                // 1 - truth
                auto one_minus_truth = make_shared<TensorMinusScalarView>(1.0f, truths[i]);
                // 1 - prediction
                auto one_minus_prediction = make_shared<TensorMinusScalarView>(1.0f, predictions[i]);

                // (1 - truth) / (1 - prediction)
                auto one_minus_truth_div_one_minus_prediction = make_shared<TensorElementWiseDivideByTensorView>(one_minus_truth, one_minus_prediction);

                // truth/prediction - (1 - truth) / (1 - prediction)
                auto unscaled_derivative = make_shared<TensorSubtractTensorView>(truth_div_prediction, one_minus_truth_div_one_minus_prediction);

                // -(truth/prediction - (1 - truth) / (1 - prediction)) / batch_size
                auto scaled_derivative = make_shared<TensorMultiplyByScalarView>(unscaled_derivative, -1.0f / batch_size);
                if (i == 0) {
                    accumulatedLossDerivative = scaled_derivative;
                } else {
                    accumulatedLossDerivative = make_shared<TensorAddTensorView>(accumulatedLossDerivative, scaled_derivative);
                }
            }
            return accumulatedLossDerivative;
        }

    };
}
#endif //HAPPYML_BINARY_CROSS_ENTROPY_HPP
