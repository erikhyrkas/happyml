//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_BINARY_CROSS_ENTROPY_HPP
#define HAPPYML_BINARY_CROSS_ENTROPY_HPP

namespace happyml {
    // binary cross entropy is -1 * average(truth *log(prediction)) + (1-truth) * log(1-pred))
// encoding might look like [1, 1, 0] where each element is 0 or 1, and we're predicting which are 1s and which are 0s
    class BinaryCrossEntropyLossFunction : public LossFunction {
        float compute(shared_ptr<BaseTensor> &total_error) override {
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
            throw exception("not implemented");
        }

        shared_ptr<BaseTensor> partialDerivative(shared_ptr<BaseTensor> &total_error, float batch_size) override {
            throw exception("not implemented");
        }
    };
}
#endif //HAPPYML_BINARY_CROSS_ENTROPY_HPP
