//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MSE_LOSS_HPP
#define HAPPYML_MSE_LOSS_HPP

namespace happyml {
    class MeanSquaredErrorLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            return make_shared<SubtractTensorView>(prediction, truth);
        }

        float compute_loss(shared_ptr<BaseTensor> &total_error) override {
            // for a single prediction: mean of squared error = avg( (prediction - truth)^2 )
            // auto error = make_shared<TensorMinusTensorView>(prediction, truth);
            // for batch, we take the average error: avg( avg(prediction - truth)^2 )
            auto squared_error = make_shared<PowerTensorView>(total_error, 2.0f);
            return squared_error->arithmeticMean(); // mean of squared error
        }

        shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                       shared_ptr<BaseTensor> &truth,
                                                       shared_ptr<BaseTensor> &prediction) override {
            // derivative of mean squared error = 2 * (prediction - truth);
            shared_ptr<BaseTensor> result = make_shared<ScalarMultiplyTensorView>(total_batch_error, 2.0f);
            return result;
        }
    };
}

#endif //HAPPYML_MSE_LOSS_HPP
