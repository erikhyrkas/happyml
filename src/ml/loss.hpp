//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_LOSS_HPP
#define HAPPYML_LOSS_HPP

#include "../types/tensor.hpp"
#include "../util/basic_profiler.hpp"

using namespace std;

namespace happyml {

    // also known as the "cost" function
    class LossFunction {
    public:
        shared_ptr<BaseTensor> calculateError(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) {
            return make_shared<TensorMinusTensorView>(prediction, truth);
        }

        shared_ptr<BaseTensor> calculateTotalError(vector<shared_ptr<BaseTensor>> &truths,
                                                   vector<shared_ptr<BaseTensor>> &predictions) {
            PROFILE_BLOCK(profileBlock);
            size_t count = truths.size();
            if (count == 1) {
                return calculateError(truths[0], predictions[0]);
            }
            shared_ptr<BaseTensor> total_error = calculateError(truths[0], predictions[0]);
            for (size_t i = 1; i < count; i++) {
                auto next_error = calculateError(truths[i], predictions[i]);
                total_error = make_shared<TensorAddTensorView>(total_error, next_error);
            }
            return total_error;
        }

        // mostly for display, but can be used for early stopping.
        virtual float compute(shared_ptr<BaseTensor> total_error) = 0;

        // what we actually use to learn
        virtual shared_ptr<BaseTensor> partialDerivative(shared_ptr<BaseTensor> total_error, float batch_size) = 0;
    };

    class MeanSquaredErrorLossFunction : public LossFunction {
        float compute(shared_ptr<BaseTensor> total_error) override {
            // for a single prediction: mean of squared error = avg( (prediction - truth)^2 )
            // auto error = make_shared<TensorMinusTensorView>(prediction, truth);
            // for batch, we take the average error: avg( avg(prediction - truth)^2 )
            auto squared_error = make_shared<TensorPowerView>(total_error, 2.0f);
            return squared_error->arithmeticMean(); // mean of squared error
        }

        shared_ptr<BaseTensor> partialDerivative(shared_ptr<BaseTensor> total_error, float batch_size) override {
            // derivative of mean squared error = 2 * (prediction - truth);
            //const auto error = make_shared<TensorMinusTensorView>(prediction, truth);
            return make_shared<TensorMultiplyByScalarView>(total_error, 2.0f / batch_size);
        }
    };

//// I used this as reference: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
//// cross entropy is sum( truth * log(prediction))
//    class CategoricalCrossEntropyLossFunction : public LossFunction {
//        float compute(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
//            // cross_entropy = sum( truth * log(prediction))
//            auto log_pred = make_shared<TensorLog2View>(prediction);
//            auto truth_dot_log_pred = make_shared<TensorMatrixMultiplyTensorView>(truth, log_pred);
//            return (float) truth_dot_log_pred->sum();
//        }
//    };
//
//// todo: sparse categorical cross entropy
//// categorical cross entropy assumes encoding like [0, 1, 0], [1, 0, 0] where sparse categorical cross entropy is [2], [4]
//
//// binary cross entropy is -1 * average(truth *log(prediction)) + (1-truth) * log(1-pred))
//// encoding might look like [1, 1, 0] where each element is 0 or 1, and we're predicting which are 1s and which are 0s
//    class BinaryCrossEntropyLossFunction : public LossFunction {
//        float compute(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
//            // binary cross_entropy = - avg( truth * log(pred) + (1-truth) * log(1-pred) )
//            auto log_pred = make_shared<TensorLog2View>(prediction);
//            auto truth_dot_log_pred = make_shared<TensorMatrixMultiplyTensorView>(truth, log_pred);
//
//            auto neg_truth = make_shared<TensorMultiplyByScalarView>(truth, -1.0);
//            auto one_minus_truth = make_shared<TensorAddScalarView>(truth, 1.0);
//
//            auto neg_pred = make_shared<TensorMultiplyByScalarView>(prediction, -1.0);
//            auto one_minus_pred = make_shared<TensorAddScalarView>(prediction, 1.0);
//            auto log_one_minus_pred = make_shared<TensorLog2View>(one_minus_pred);
//
//            auto one_minus_truth_dot_log_one_minus_pred = make_shared<TensorMatrixMultiplyTensorView>(one_minus_truth,
//                                                                                           log_one_minus_pred);
//
//            auto result_tensor = make_shared<TensorAddTensorView>(truth_dot_log_pred,
//                                                                  one_minus_truth_dot_log_one_minus_pred);
//            return -result_tensor->arithmetic_mean();
//        }
//    };
//
////categorical_crossentropy


    shared_ptr<LossFunction> createLoss(LossType lossType) {
        shared_ptr<LossFunction> lossFunction;
        switch (lossType) {
            case LossType::mse:
                lossFunction = make_shared<MeanSquaredErrorLossFunction>();
                break;
            default:
                lossFunction = make_shared<MeanSquaredErrorLossFunction>();
        }
        return lossFunction;
    }
}
#endif //HAPPYML_LOSS_HPP
