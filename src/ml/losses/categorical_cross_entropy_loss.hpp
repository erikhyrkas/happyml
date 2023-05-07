//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include <cmath>
#include "../../types/tensor_views/tensor_log_view.hpp"
#include "../../types/tensor_views/tensor_element_wise_divide_by_tensor_view.hpp"
#include "../../types/tensor_views/tensor_exp_view.hpp"
#include "../../types/tensor_views/tensor_value_transform_view.hpp"

// TODO: this code currently assumes that the truth and predictions are both 1D tensors
//  and that those tensors are 1 row and 1 channel.

namespace happyml {
    // The CategoricalCrossEntropyLossFunction class implements the categorical cross-entropy loss function for
// multi-class classification problems. It computes the loss and its derivative with respect to the
// model's predictions.
    class CategoricalCrossEntropyLossFunction : public LossFunction {
    public:
        // calculate_error_for_one_prediction computes the element-wise error for a single prediction
        // using the categorical cross-entropy formula: -truth_i * log(prediction_i)
        shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Clip the prediction tensor to avoid log(0) and log(1) edge cases
            auto epsilon = 1e-8f;
            auto clip_prediction = make_shared<TensorClipView>(prediction, epsilon, 1.0f - epsilon);

            // Compute -truth
            auto negative_truth = make_shared<TensorMultiplyByScalarView>(truth, -1.0f);

            // Compute log(prediction)
            auto log_pred = make_shared<TensorLogView>(clip_prediction);

            // Compute -truth * log(prediction)
            auto neg_truth_by_log_pred = make_shared<TensorElementWiseMultiplyByTensorView>(negative_truth, log_pred);

            return neg_truth_by_log_pred;
        }

        // computeBatchLoss computes the average loss for a batch of predictions
        float computeBatchLoss(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: -truth_i * log(prediction_i)
            // For a single prediction: categorical cross-entropy = sum(total_error)
            // For a batch, we take the average error: avg(sum(total_error))
            return total_error->arithmeticMean();
        }

        // calculate_batch_loss_derivative computes the derivative of the categorical cross-entropy loss
        // with respect to the predictions for a batch of examples
        shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               vector<shared_ptr<BaseTensor>> &truths,
                                                               vector<shared_ptr<BaseTensor>> &predictions) override {
            size_t batch_size = truths.size();
            // Initialize the accumulated loss derivative tensor to zero
            shared_ptr<BaseTensor> accumulatedLossDerivative = make_shared<UniformTensor>(predictions[0]->getShape(), 0.0f);

            for (size_t i = 0; i < batch_size; i++) {
                // Compute the loss derivative for the current example: prediction_i - truth_i
                shared_ptr<BaseTensor> currentLossDerivative = make_shared<TensorSubtractTensorView>(predictions[i], truths[i]);

                // Accumulate the loss derivatives for all examples in the batch
                accumulatedLossDerivative = make_shared<TensorAddTensorView>(accumulatedLossDerivative, currentLossDerivative);
            }

            // Average the accumulated loss derivative over the batch size
            auto result = make_shared<TensorMultiplyByScalarView>(accumulatedLossDerivative, 1.0f / (float) batch_size);

            return result;
        }
    };
}
#endif //HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
