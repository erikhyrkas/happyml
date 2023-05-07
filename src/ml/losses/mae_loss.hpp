//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_MAE_LOSS_HPP
#define HAPPYML_MAE_LOSS_HPP

#include "../../types/tensor_views/tensor_absolute_view.hpp"

namespace happyml {
    // Also known as L1 loss
    // Can be useful for regression tasks.
    class MeanAbsoluteErrorLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> calculateError(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as: |prediction_i - truth_i|
            shared_ptr<BaseTensor> error = make_shared<TensorAbsoluteView>(make_shared<TensorMinusTensorView>(prediction, truth));
            return error;
        }

        float compute(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: |prediction_i - truth_i|
            // For a single prediction: mean absolute error = sum(total_error) / number of elements
            // For a batch, we take the average error: avg(sum(total_error) / number of elements)
            return total_error->arithmeticMean();
        }

        pair<shared_ptr<BaseTensor>, shared_ptr<BaseTensor>> calculateBatchErrorAndDerivative(vector<shared_ptr<BaseTensor>> &truths,
                                                                                              vector<shared_ptr<BaseTensor>> &predictions) override {
            shared_ptr<BaseTensor> totalError = calculateTotalError(truths, predictions);

            shared_ptr<BaseTensor> accumulatedLossDerivative;
            for (size_t i = 0; i < truths.size(); i++) {
                // Derivative of mean absolute error = sign(prediction_i - truth_i)
                shared_ptr<BaseTensor> error_diff = make_shared<TensorMinusTensorView>(predictions[i], truths[i]);
                shared_ptr<BaseTensor> currentLossDerivative = make_shared<TensorMultiplyByScalarView>(error_diff, -1.0f);
                if (i == 0) {
                    accumulatedLossDerivative = currentLossDerivative;
                } else {
                    accumulatedLossDerivative = make_shared<TensorAddTensorView>(accumulatedLossDerivative, currentLossDerivative);
                }
            }

            // Divide the accumulated derivative by the batch size to get the average derivative
            auto batch_size = static_cast<float>(truths.size());
            shared_ptr<BaseTensor> averagedLossDerivative = make_shared<TensorMultiplyByScalarView>(accumulatedLossDerivative, 1.0f / batch_size);

            return make_pair(totalError, averagedLossDerivative);
        }
    };
}
#endif //HAPPYML_MAE_LOSS_HPP
