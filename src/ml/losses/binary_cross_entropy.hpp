//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_BINARY_CROSS_ENTROPY_HPP
#define HAPPYML_BINARY_CROSS_ENTROPY_HPP

#include "../../types/tensor_views/tensor_minus_tensor_view.hpp"
#include "../../types/tensor_views/tensor_minus_scalar_view.hpp"
#include "../../types/tensor_views/tensor_matrix_divide_tensor_view.hpp"
#include "../../types/tensor_views/tensor_multiply_tensor_view.hpp"
#include "../../types/tensor_views/tensor_matrix_multiply_tensor_view.hpp"

namespace happyml {
    // binary cross entropy is -1 * average(truth *log(prediction)) + (1-truth) * log(1-pred))
    // encoding might look like [1, 1, 0] where each element is 0 or 1, and we're predicting which are 1s and which are 0s
    class BinaryCrossEntropyLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> calculateError(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            //     float loss = 0.0;
            //    for (size_t i = 0; i < y_pred.size(); i++) {
            //        loss -= y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
            //    }
            //    return loss;
            auto epsilon = 1e-8f;
            auto clip_prediction = make_shared<TensorClipView>(prediction, epsilon, 1.0f - epsilon);
            auto clip_one_minus_prediction = make_shared<TensorClipView>(make_shared<TensorMinusScalarView>(1.0f, prediction), epsilon, 1.0f - epsilon);

            auto log_prediction = make_shared<TensorLogView>(clip_prediction);
            auto log_one_minus_prediction = make_shared<TensorLogView>(clip_one_minus_prediction);

            auto one_minus_truth = make_shared<TensorMinusScalarView>(1.0f, truth);

            auto truth_matmul_log_prediction = make_shared<TensorMatrixMultiplyTensorView>(truth, log_prediction);
            auto one_minus_truth_matmul_log_one_minus_prediction = make_shared<TensorMatrixMultiplyTensorView>(one_minus_truth, log_one_minus_prediction);

            auto total_error = make_shared<TensorAddTensorView>(truth_matmul_log_prediction, one_minus_truth_matmul_log_one_minus_prediction);
            auto negative_total_error = make_shared<TensorMultiplyByScalarView>(total_error, -1.0f);
            return negative_total_error;
        }


        float compute(shared_ptr<BaseTensor> &total_error) override {
            // The total_error tensor is precomputed as: -truth_i * log(prediction_i) - (1 - truth_i) * log(1 - prediction_i)
            // For a single prediction: binary cross-entropy = sum(total_error)
            // For a batch, we take the average error: avg(sum(total_error))
            auto batch_size = static_cast<float>(total_error->size());
            auto sum_error = total_error->sum();
            auto avg_error = sum_error / batch_size;
            return static_cast<float>(avg_error);
        }

        pair<shared_ptr<BaseTensor>, shared_ptr<BaseTensor>> calculateBatchErrorAndDerivative(vector<shared_ptr<BaseTensor>> &truths,
                                                                                              vector<shared_ptr<BaseTensor>> &predictions) override {
            shared_ptr<BaseTensor> total_error = calculateTotalError(truths, predictions);

            // batch_size should be equal to the size of truths or predictions
            auto batch_size = static_cast<float>(truths.size());

            shared_ptr<BaseTensor> accumulatedLossDerivative;
            for (size_t i = 0; i < truths.size(); i++) {
                // Derivative of binary cross-entropy = (prediction - truth) / (prediction * (1 - prediction)) / batch_size

                //prediction - truth
                auto error_diff = make_shared<TensorMinusTensorView>(predictions[i], truths[i]);

                // 1 - prediction
                auto negative_prediction = make_shared<TensorMinusScalarView>(1.0f, predictions[i]);
                // prediction * (1 - prediction)
                auto inverse_denominator = make_shared<TensorMultiplyTensorView>(predictions[i], negative_prediction);
                // (prediction - truth) / (prediction * (1 - prediction))
                auto unscaled_derivative = make_shared<TensorMatrixDivideTensorView>(error_diff, inverse_denominator);
                // (prediction - truth) / (prediction * (1 - prediction)) / batch_size
                auto scaled_derivative = make_shared<TensorMultiplyByScalarView>(unscaled_derivative, 1.0f / batch_size);

                if (i == 0) {
                    accumulatedLossDerivative = scaled_derivative;
                } else {
                    accumulatedLossDerivative = make_shared<TensorAddTensorView>(accumulatedLossDerivative, scaled_derivative);
                }
            }
            return {total_error, accumulatedLossDerivative};
        }
    };

}
#endif //HAPPYML_BINARY_CROSS_ENTROPY_HPP
