//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_MAE_LOSS_HPP
#define HAPPYML_MAE_LOSS_HPP

#include "../../types/tensor_views/absolute_tensor_view.hpp"

namespace happyml {
    // Also known as L1 loss
    // Can be useful for regression tasks.
    class MeanAbsoluteErrorLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as: |prediction_i - truth_i|
            shared_ptr<BaseTensor> error = make_shared<AbsoluteTensorView>(make_shared<SubtractTensorView>(prediction, truth));
            return error;
        }

        float compute_loss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: |prediction_i - truth_i|
            // For a single prediction: mean absolute error = sum(total_error) / number of elements
            // For a batch, we take the average error: avg(sum(total_error) / number of elements)
            return total_error->arithmeticMean();
        }

        shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                       shared_ptr<BaseTensor> &truth,
                                                       shared_ptr<BaseTensor> &prediction) override {
            shared_ptr<BaseTensor> accumulatedLossDerivative;
            // Derivative of mean absolute error = sign(prediction_i - truth_i)
            shared_ptr<BaseTensor> error_diff = make_shared<SubtractTensorView>(prediction, truth);
            shared_ptr<BaseTensor> currentLossDerivative = make_shared<ScalarMultiplyTensorView>(error_diff, -1.0f);
            return currentLossDerivative;
        }
    };
}
#endif //HAPPYML_MAE_LOSS_HPP
