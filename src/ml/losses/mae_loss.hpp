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
        shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as: |prediction_i - truth_i|
            shared_ptr<BaseTensor> error = make_shared<AbsoluteTensorView>(make_shared<SubtractTensorView>(prediction, truth));
            return error;
        }

        float computeBatchLoss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: |prediction_i - truth_i|
            // For a single prediction: mean absolute error = sum(total_error) / number of elements
            // For a batch, we take the average error: avg(sum(total_error) / number of elements)
            return total_error->arithmeticMean();
        }

        shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               vector<shared_ptr<BaseTensor>> &truths,
                                                               vector<shared_ptr<BaseTensor>> &predictions) override {
            shared_ptr<BaseTensor> accumulatedLossDerivative;
            for (size_t i = 0; i < truths.size(); i++) {
                // Derivative of mean absolute error = sign(prediction_i - truth_i)
                shared_ptr<BaseTensor> error_diff = make_shared<SubtractTensorView>(predictions[i], truths[i]);
                shared_ptr<BaseTensor> currentLossDerivative = make_shared<ScalarMultiplyTensorView>(error_diff, -1.0f);
                if (i == 0) {
                    accumulatedLossDerivative = currentLossDerivative;
                } else {
                    accumulatedLossDerivative = make_shared<AddTensorView>(accumulatedLossDerivative, currentLossDerivative);
                }
            }

            // Divide the accumulated derivative by the batch size to get the average derivative
            auto batch_size = static_cast<float>(truths.size());
            shared_ptr<BaseTensor> averagedLossDerivative = make_shared<ScalarMultiplyTensorView>(accumulatedLossDerivative, 1.0f / batch_size);

            return averagedLossDerivative;
        }
    };
}
#endif //HAPPYML_MAE_LOSS_HPP
