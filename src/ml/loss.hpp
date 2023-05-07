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
        shared_ptr<BaseTensor> sum_total_batch_error(vector<shared_ptr<BaseTensor>> &truths,
                                                     vector<shared_ptr<BaseTensor>> &predictions) {
            PROFILE_BLOCK(profileBlock);
            size_t count = truths.size();
            if (count == 1) {
                return calculate_error_for_one_prediction(truths[0], predictions[0]);
            }
            auto total_error = calculate_error_for_one_prediction(truths[0], predictions[0]);
            for (size_t i = 1; i < count; i++) {
                auto next_error = calculate_error_for_one_prediction(truths[i], predictions[i]);
                total_error = make_shared<TensorAddTensorView>(total_error, next_error);
            }
            return make_shared<FullTensor>(total_error);
        }

        virtual shared_ptr<BaseTensor> calculate_error_for_one_prediction(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) = 0;

        // mostly for display, but can be used for early stopping.
        virtual float computeBatchLoss(shared_ptr<BaseTensor> &total_batch_error) = 0;

        virtual shared_ptr<BaseTensor> calculate_batch_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                                       vector<shared_ptr<BaseTensor>> &truths,
                                                                       vector<shared_ptr<BaseTensor>> &predictions) = 0;

    };


}
#endif //HAPPYML_LOSS_HPP
