//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_SMAE_LOSS_HPP
#define HAPPYML_SMAE_LOSS_HPP

#include "../../types/tensor_views/tensor_less_than_scalar_view.hpp"
#include "../../types/tensor_views/tensor_masked_select_view.hpp"

namespace happyml {
    // Also known as Smooth L1 loss, Smooth Mean Absolute Error loss, or Huber loss
    // Combines the benefits of both Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    // It is less sensitive to outliers than MSE and is smoother than MAE.
    // Huber Loss is often used in robust regression problems.
    class SmoothMeanAbsoluteErrorLossFunction : public LossFunction {
    public:
        float smoothness = 1.0f; // point where it changes from a quadratic to linear function.

        shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as:
            // if |prediction_i - truth_i| < smoothness, 0.5 * (prediction_i - truth_i)^2 / smoothness
            // else, |prediction_i - truth_i| - 0.5 * smoothness
            auto error_diff = make_shared<TensorSubtractTensorView>(prediction, truth);
            auto abs_error_diff = make_shared<TensorAbsoluteView>(error_diff);
            auto squared_error_diff = make_shared<TensorPowerView>(error_diff, 2.0f);

            auto smooth_part = make_shared<TensorMultiplyByScalarView>(squared_error_diff, 0.5f / smoothness);
            auto unsmooth_part = make_shared<TensorMinusScalarView>(abs_error_diff, 0.5f * smoothness);

            auto smooth_mask = make_shared<TensorLessThanScalarView>(abs_error_diff, smoothness);
            auto error = make_shared<TensorMaskedSelectView>(smooth_mask, smooth_part, unsmooth_part);
            return error;
        }

        float computeBatchLoss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as a combination of smooth and unsmooth parts
            // For a single prediction: smooth mean absolute error = sum(total_error) / number of elements
            // For a batch, we take the average error: avg(sum(total_error) / number of elements)
            auto sum_error = (float) total_error->sum();
            return sum_error / (float) total_error->size();
        }

        shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               vector<shared_ptr<BaseTensor>> &truths,
                                                               vector<shared_ptr<BaseTensor>> &predictions) override {
            shared_ptr<BaseTensor> accumulatedLossDerivative;
            for (size_t i = 0; i < truths.size(); i++) {
                // Derivative of smooth mean absolute error:
                // if |prediction_i - truth_i| < smoothness, (prediction_i - truth_i) / smoothness
                // else, sign(prediction_i - truth_i)
                shared_ptr<BaseTensor> error_diff = make_shared<TensorSubtractTensorView>(predictions[i], truths[i]);
                shared_ptr<BaseTensor> abs_error_diff = make_shared<TensorAbsoluteView>(error_diff);
                shared_ptr<BaseTensor> smooth_derivative = make_shared<TensorMultiplyByScalarView>(error_diff, 1.0f / smoothness);
                shared_ptr<BaseTensor> unsmooth_derivative = make_shared<TensorMultiplyByScalarView>(error_diff, -1.0f);

                shared_ptr<BaseTensor> smooth_mask = make_shared<TensorLessThanScalarView>(abs_error_diff, smoothness);
                shared_ptr<BaseTensor> currentLossDerivative = make_shared<TensorMaskedSelectView>(smooth_mask, smooth_derivative, unsmooth_derivative);

                if (i == 0) {
                    accumulatedLossDerivative = currentLossDerivative;
                } else {
                    accumulatedLossDerivative = make_shared<TensorAddTensorView>(accumulatedLossDerivative, currentLossDerivative);
                }
            }

            return accumulatedLossDerivative;
        }
    };
}
#endif //HAPPYML_SMAE_LOSS_HPP
