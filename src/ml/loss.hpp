//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_LOSS_HPP
#define HAPPYML_LOSS_HPP

#include "../util/basic_profiler.hpp"

using namespace std;

namespace happyml {

    // also known as the "cost" function
    class LossFunction {
    public:
        virtual shared_ptr<BaseTensor> calculateError(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) {
            return make_shared<TensorMinusTensorView>(prediction, truth);
        }

        virtual shared_ptr<BaseTensor> calculateTotalError(vector<shared_ptr<BaseTensor>> &truths,
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
        virtual float compute(shared_ptr<BaseTensor> &total_error) = 0;

        // what we actually use to learn
        virtual shared_ptr<BaseTensor> partialDerivative(shared_ptr<BaseTensor> &total_error, float batch_size) = 0;

    };


}
#endif //HAPPYML_LOSS_HPP
