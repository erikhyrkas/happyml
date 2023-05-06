//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MSE_LOSS_HPP
#define HAPPYML_MSE_LOSS_HPP

#include "../loss.hpp"
#include "../../types/base_tensors.hpp"
#include "../../types/tensor_views/tensor_power_view.hpp"
#include "../../types/tensor_views/tensor_multiply_by_scalar_view.hpp"

namespace happyml {
    class MeanSquaredErrorLossFunction : public LossFunction {
    public:
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
}

#endif //HAPPYML_MSE_LOSS_HPP
