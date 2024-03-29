//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_OPTIMIZER_FACTORY_HPP
#define HAPPYML_OPTIMIZER_FACTORY_HPP

#include "enums.hpp"
#include "optimizers/mbgd_optimizer.hpp"
#include "optimizers/adam_optimizer.hpp"
#include "optimizers/sgdm_optimizer.hpp"

using namespace happyml;

namespace happyml {

    shared_ptr<BaseOptimizer> createOptimizer(const OptimizerType optimizerType,
                                              const float learningRate,
                                              const float biasLearningRate) {
        shared_ptr<BaseOptimizer> result;
        switch (optimizerType) {
            case OptimizerType::sgd:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
                break;
            case OptimizerType::adam_with_decaying_momentum:
                result = make_shared<AdamOptimizer>(learningRate, biasLearningRate, true);
                break;
            case OptimizerType::adam:
                result = make_shared<AdamOptimizer>(learningRate, biasLearningRate, false);
                break;
            case OptimizerType::sgdm_with_decaying_momentum:
                result = make_shared<SGDMOptimizer>(learningRate, biasLearningRate, true);
                break;
            case OptimizerType::sgdm:
                result = make_shared<SGDMOptimizer>(learningRate, biasLearningRate, false);
                break;
            default:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
        }
        return result;
    }
}
#endif // HAPPYML_OPTIMIZER_FACTORY_HPP