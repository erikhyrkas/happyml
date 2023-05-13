//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MSE_LOSS_HPP
#define HAPPYML_MSE_LOSS_HPP

namespace happyml {
    class MeanSquaredErrorLossFunction : public LossFunction {
    public:
        shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) override {
            return make_shared<SubtractTensorView>(prediction, truth);
        }

        float computeBatchLoss(shared_ptr<BaseTensor> &total_error) override {
            // for a single prediction: mean of squared error = avg( (prediction - truth)^2 )
            // auto error = make_shared<TensorMinusTensorView>(prediction, truth);
            // for batch, we take the average error: avg( avg(prediction - truth)^2 )
            auto squared_error = make_shared<PowerTensorView>(total_error, 2.0f);
            return squared_error->arithmeticMean(); // mean of squared error
        }

        shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               vector<shared_ptr<BaseTensor>> &truths,
                                                               vector<shared_ptr<BaseTensor>> &predictions) override {
            auto batchSize = static_cast<float>(truths.size());
            // derivative of mean squared error = 2 * (prediction - truth);
            shared_ptr<BaseTensor> result = make_shared<ScalarMultiplyTensorView>(total_batch_error, 2.0f / batchSize);
            return result;
        }
    };
}

#endif //HAPPYML_MSE_LOSS_HPP
