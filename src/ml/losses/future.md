
I moved the work-in-progress code to this file. I don't think it works and I need to sort it out later.


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

