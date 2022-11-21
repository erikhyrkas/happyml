//
// Created by Erik Hyrkas on 11/5/2022.
//

#ifndef MICROML_LOSS_HPP
#define MICROML_LOSS_HPP

#include "tensor.hpp"

class LossFunction {
public:
    // placeholder
    virtual float compute(std::shared_ptr<BaseTensor> &truth,  std::shared_ptr<BaseTensor> &prediction) = 0;
};

// TODO: is this right?
class MeanSquaredErrorLossFunction : public LossFunction {
    float compute(std::shared_ptr<BaseTensor> &truth,  std::shared_ptr<BaseTensor> &prediction) override {
        // mean of squared error = avg( (truth - prediction)^2 )
        auto error = std::make_shared<TensorMinusTensorView>(truth, prediction);
        auto squared_error = std::make_shared<TensorPowerView>(error, 2.0f);
        return squared_error->arithmetic_mean(); // mean of squared error
    }
};

// I used this as reference: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
// cross entropy is sum( truth * log(prediction))
class CategoricalCrossEntropyLossFunction : public LossFunction {
    float compute(std::shared_ptr<BaseTensor> &truth,  std::shared_ptr<BaseTensor> &prediction) override {
        // cross_entropy = sum( truth * log(prediction))
        auto log_pred = std::make_shared<TensorLog2View>(prediction);
        auto truth_dot_log_pred = std::make_shared<TensorDotTensorView>(truth, log_pred);
        return truth_dot_log_pred->sum();
    }
};

// todo: sparse categorical cross entropy
// categorical cross entropy assumes encoding like [0, 1, 0], [1, 0, 0] where sparse categorical cross entropy is [2], [4]

// binary cross entropy is -1 * average(truth *log(prediction)) + (1-truth) * log(1-pred))
// encoding might look like [1, 1, 0] where each element is 0 or 1, and we're predicting which are 1s and which are 0s
class BinaryCrossEntropyLossFunction : public LossFunction {
    float compute(std::shared_ptr<BaseTensor> &truth,  std::shared_ptr<BaseTensor> &prediction) override {
        // binary cross_entropy = - avg( truth * log(pred) + (1-truth) * log(1-pred) )
        auto log_pred = std::make_shared<TensorLog2View>(prediction);
        auto truth_dot_log_pred = std::make_shared<TensorDotTensorView>(truth, log_pred);

        auto neg_truth = std::make_shared<TensorMultiplyByScalarView>(truth, -1.0);
        auto one_minus_truth = std::make_shared<TensorAddScalarView>(truth, 1.0);

        auto neg_pred = std::make_shared<TensorMultiplyByScalarView>(prediction, -1.0);
        auto one_minus_pred = std::make_shared<TensorAddScalarView>(prediction, 1.0);
        auto log_one_minus_pred = std::make_shared<TensorLog2View>(one_minus_pred);

        auto one_minus_truth_dot_log_one_minus_pred = std::make_shared<TensorDotTensorView>(one_minus_truth, log_one_minus_pred);

        auto result_tensor = std::make_shared<TensorAddTensorView>(truth_dot_log_pred, one_minus_truth_dot_log_one_minus_pred);
        return - result_tensor->arithmetic_mean();
    }
};

//categorical_crossentropy

#endif //MICROML_LOSS_HPP
