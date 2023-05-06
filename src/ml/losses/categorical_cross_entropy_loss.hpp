//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include <cmath>
#include "../../types/tensor_views/tensor_log_view.hpp"

// FIXME: This code isn't finished.

namespace happyml {
// I used this as reference: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
// cross entropy is sum( truth * log(prediction))
//            cross_entropy = sum( truth * log(prediction))
//            auto log_pred = make_shared<TensorLog2View>(prediction);
//            auto truth_dot_log_pred = make_shared<TensorMatrixMultiplyTensorView>(truth, log_pred);
//            return (float) truth_dot_log_pred->sum();

    class CategoricalCrossEntropyLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> calculateError(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            // Calculate per-element error as: -truth_i * log(prediction_i)
            auto log_pred = make_shared<TensorLogView>(prediction);
            auto negative_truth = make_shared<TensorMultiplyByScalarView>(truth, -1.0f);
            auto truth_dot_log_pred = make_shared<TensorMultiplyTensorView>(negative_truth, log_pred);
            return truth_dot_log_pred;
        }

        float compute(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: -truth_i * log(prediction_i)
            // For a single prediction: categorical cross-entropy = sum(total_error)
            // For a batch, we take the average error: avg(sum(total_error))

            float sum_error = total_error->sum();
            return sum_error / total_error->size();
        }

        shared_ptr<BaseTensor> partialDerivative(shared_ptr<BaseTensor> &total_error, float batch_size) override {
            // Derivative of categorical cross-entropy = -truth_i / prediction_i
//            auto negative_truth = make_shared<TensorMultiplyByScalarView>(truth, -1.0f);
//            auto inverse_prediction = make_shared<TensorInverseView>(prediction);
//            auto matmul_result = make_shared<TensorMultiplyTensorView>(negative_truth, inverse_prediction);
//            return matmul_result;
            throw exception("Not implemented");
        }
    };

}
#endif //HAPPYML_CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
