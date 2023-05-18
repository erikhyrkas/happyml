//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_BINARY_CROSS_ENTROPY_HPP
#define HAPPYML_BINARY_CROSS_ENTROPY_HPP

#include "../../types/tensor_views/scalar_subtract_tensor_view.hpp"
#include "../../types/tensor_views/matrix_divide_tensor_view.hpp"

namespace happyml {
    // The BinaryCrossEntropyLossFunction class implements the binary cross-entropy loss function for
    // binary classification problems. It computes the loss and its derivative with respect to the
    // model's predictions.
    class BinaryCrossEntropyLossFunction : public LossFunction {
    public:
        // calculate_error_for_one_prediction computes the element-wise error for a single prediction
        // using the binary cross-entropy formula:
        // -truth * log(prediction) - (1 - truth) * log(1 - prediction)
        shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            auto epsilon = 1e-8f;
            auto clip_prediction = make_shared<ClipTensorView>(prediction, epsilon, 1.0f - epsilon);
            auto clip_one_minus_prediction = make_shared<ClipTensorView>(make_shared<ScalarSubtractTensorView>(1.0f, prediction), epsilon, 1.0f - epsilon);

            auto log_prediction = make_shared<LogTensorView>(clip_prediction);
            auto log_one_minus_prediction = make_shared<LogTensorView>(clip_one_minus_prediction);

            auto one_minus_truth = make_shared<ScalarSubtractTensorView>(1.0f, truth);

            auto truth_matmul_log_prediction = make_shared<ElementWiseMultiplyTensorView>(truth, log_prediction);
            auto one_minus_truth_matmul_log_one_minus_prediction = make_shared<ElementWiseMultiplyTensorView>(one_minus_truth, log_one_minus_prediction);

            auto total_error = make_shared<AddTensorView>(truth_matmul_log_prediction, one_minus_truth_matmul_log_one_minus_prediction);
            auto negative_total_error = make_shared<ScalarMultiplyTensorView>(total_error, -1.0f);
            return negative_total_error;
        }

        // computeBatchLoss computes the average loss for a batch of predictions
        float compute_loss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: -truth_i * log(prediction_i) - (1 - truth_i) * log(1 - prediction_i)
            // For a single prediction: binary cross-entropy = sum(total_error)
            auto sum_error = (float) total_error->sum();
            return sum_error;
        }

        // calculate_batch_loss_derivative computes the derivative of the binary cross-entropy loss
        // with respect to the predictions for a batch of examples
        shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                       shared_ptr<BaseTensor> &truth,
                                                       shared_ptr<BaseTensor> &prediction) override {
            // Derivative of binary cross-entropy = -(truth/prediction - (1 - truth) / (1 - prediction))

            // truth / prediction
            auto truth_div_prediction = make_shared<ElementWiseDivideTensorView>(truth, prediction);

            // 1 - truth
            auto one_minus_truth = make_shared<ScalarSubtractTensorView>(1.0f, truth);
            // 1 - prediction
            auto one_minus_prediction = make_shared<ScalarSubtractTensorView>(1.0f, prediction);

            // (1 - truth) / (1 - prediction)
            auto one_minus_truth_div_one_minus_prediction = make_shared<ElementWiseDivideTensorView>(one_minus_truth, one_minus_prediction);

            // truth/prediction - (1 - truth) / (1 - prediction)
            auto unscaled_derivative = make_shared<SubtractTensorView>(truth_div_prediction, one_minus_truth_div_one_minus_prediction);

            // -(truth/prediction - (1 - truth) / (1 - prediction)) / batch_size
            auto scaled_derivative = make_shared<ScalarMultiplyTensorView>(unscaled_derivative, -1.0f);
            return scaled_derivative;
        }

    };
}
#endif //HAPPYML_BINARY_CROSS_ENTROPY_HPP
