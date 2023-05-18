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

        virtual shared_ptr<BaseTensor> compute_error(shared_ptr<BaseTensor> &truth, shared_ptr<BaseTensor> &prediction) = 0;

        // mostly for display, but can be used for early stopping.
        virtual float compute_loss(shared_ptr<BaseTensor> &error) = 0;

        virtual shared_ptr<BaseTensor> compute_loss_derivative(shared_ptr<BaseTensor> &total_batch_error,
                                                               shared_ptr<BaseTensor> &truth,
                                                               shared_ptr<BaseTensor> &prediction) = 0;

    };


}
#endif //HAPPYML_LOSS_HPP
