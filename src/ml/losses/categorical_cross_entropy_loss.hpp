//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include <cmath>
#include "../../types/tensor_views/log_tensor_view.hpp"
#include "../../types/tensor_views/element_wise_divide_tensor_view.hpp"
#include "../../types/tensor_views/exponential_tensor_view.hpp"
#include "../../types/tensor_views/value_transform_tensor_view.hpp"
#include "../../types/tensor_views/scalar_divide_tensor_view.hpp"

// TODO: this code currently assumes that the truth and predictions are both 1D tensors
//  and that those tensors are 1 row and 1 channel.

namespace happyml {
    // The CategoricalCrossEntropyLossFunction class implements the categorical cross-entropy loss function for
    // multi-class classification problems. It computes the loss and its derivative with respect to the
    // model's predictions.
    class CategoricalCrossEntropyLossFunction : public LossFunction {
    public:
        // compute_error computes the element-wise error for a single prediction
        // using the categorical cross-entropy formula: -truth_i * log(prediction_i)
        shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Compute -truth
            auto negative_truth = make_shared<ScalarMultiplyTensorView>(truth, -1.0f);

            // Compute log(prediction)
            // note that LogTensorView clips the prediction to the range [1e-8, 1.0 - 1e-8]
            auto log_pred = make_shared<LogTensorView>(prediction);

            // Compute -truth * log(prediction)
            auto neg_truth_by_log_pred = make_shared<ElementWiseMultiplyTensorView>(negative_truth, log_pred);

            return neg_truth_by_log_pred;
        }

        // compute_loss computes the loss for a single prediction
        float compute_loss(shared_ptr<BaseTensor> &error) override {
            // The error tensor is precomputed as: -truth_i * log(prediction_i)
            // For a single prediction: categorical cross-entropy = sum(total_error)
            return (float) error->sum();
        }

        // calculate_batch_loss_derivative computes the derivative of the categorical cross-entropy loss
        // with respect to the predictions for a batch of examples
        shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                       shared_ptr<BaseTensor> &truth,
                                                       shared_ptr<BaseTensor> &prediction) override {
            // The total_batch_error tensor is precomputed as: -truth_i * log(prediction_i)
            auto error = make_shared<SubtractTensorView>(prediction, truth);
            return error;
        }
    };
}
#endif //HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
