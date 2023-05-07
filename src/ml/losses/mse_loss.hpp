//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MSE_LOSS_HPP
#define HAPPYML_MSE_LOSS_HPP

namespace happyml {
    class MeanSquaredErrorLossFunction : public LossFunction {
    public:
        float compute(shared_ptr<BaseTensor> &total_error) override {
            // for a single prediction: mean of squared error = avg( (prediction - truth)^2 )
            // auto error = make_shared<TensorMinusTensorView>(prediction, truth);
            // for batch, we take the average error: avg( avg(prediction - truth)^2 )
            auto squared_error = make_shared<TensorPowerView>(total_error, 2.0f);
            return squared_error->arithmeticMean(); // mean of squared error
        }

        pair<shared_ptr<BaseTensor>, shared_ptr<BaseTensor>> calculateBatchErrorAndDerivative(vector<shared_ptr<BaseTensor>> &truths,
                                                                                              vector<shared_ptr<BaseTensor>> &predictions) override {
            shared_ptr<BaseTensor> totalError = calculateTotalError(truths, predictions);
            auto batchSize =  static_cast<float>(truths.size());
            // derivative of mean squared error = 2 * (prediction - truth);
            shared_ptr<BaseTensor> result = make_shared<TensorMultiplyByScalarView>(totalError, 2.0f / batchSize);
            return make_pair(totalError, result);
        }
    };
}

#endif //HAPPYML_MSE_LOSS_HPP
