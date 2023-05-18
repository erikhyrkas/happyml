//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_SMAE_LOSS_HPP
#define HAPPYML_SMAE_LOSS_HPP

#include "../../types/tensor_views/less_than_scalar_tensor_view.hpp"
#include "../../types/tensor_views/masked_select_tensor_view.hpp"

namespace happyml {
    // Also known as Smooth L1 loss, Smooth Mean Absolute Error loss, or Huber loss
    // Combines the benefits of both Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    // It is less sensitive to outliers than MSE and is smoother than MAE.
    // Huber Loss is often used in robust regression problems.
    class SmoothMeanAbsoluteErrorLossFunction : public LossFunction {
    public:
        float smoothness = 1.0f; // point where it changes from a quadratic to linear function.

        shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as:
            // if |prediction_i - truth_i| < smoothness, 0.5 * (prediction_i - truth_i)^2 / smoothness
            // else, |prediction_i - truth_i| - 0.5 * smoothness
            auto error_diff = make_shared<SubtractTensorView>(prediction, truth);
            auto abs_error_diff = make_shared<AbsoluteTensorView>(error_diff);
            auto squared_error_diff = make_shared<PowerTensorView>(error_diff, 2.0f);

            auto smooth_part = make_shared<ScalarMultiplyTensorView>(squared_error_diff, 0.5f / smoothness);
            auto unsmooth_part = make_shared<ScalarSubtractTensorView>(abs_error_diff, 0.5f * smoothness);

            auto smooth_mask = make_shared<LessThanScalarTensorView>(abs_error_diff, smoothness);
            auto error = make_shared<MaskedSelectTensorView>(smooth_mask, smooth_part, unsmooth_part);
            return error;
        }

        float compute_loss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as a combination of smooth and unsmooth parts
            // For a single prediction: smooth mean absolute error = sum(total_error) / number of elements
            // For a batch, we take the average error: avg(sum(total_error) / number of elements)
            auto sum_error = (float) total_error->sum();
            return sum_error / (float) total_error->size();
        }

        shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                       shared_ptr<BaseTensor> &truth,
                                                       shared_ptr<BaseTensor> &prediction) override {
            // Derivative of smooth mean absolute error:
            // if |prediction_i - truth_i| < smoothness, (prediction_i - truth_i) / smoothness
            // else, sign(prediction_i - truth_i)
            shared_ptr<BaseTensor> error_diff = make_shared<SubtractTensorView>(prediction, truth);
            shared_ptr<BaseTensor> abs_error_diff = make_shared<AbsoluteTensorView>(error_diff);
            shared_ptr<BaseTensor> smooth_derivative = make_shared<ScalarMultiplyTensorView>(error_diff, 1.0f / smoothness);
            shared_ptr<BaseTensor> unsmooth_derivative = make_shared<ScalarMultiplyTensorView>(error_diff, -1.0f);

            shared_ptr<BaseTensor> smooth_mask = make_shared<LessThanScalarTensorView>(abs_error_diff, smoothness);
            shared_ptr<BaseTensor> currentLossDerivative = make_shared<MaskedSelectTensorView>(smooth_mask, smooth_derivative, unsmooth_derivative);
            return currentLossDerivative;
        }
    };
}
#endif //HAPPYML_SMAE_LOSS_HPP
