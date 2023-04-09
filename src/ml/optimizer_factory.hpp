//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//
#ifndef HAPPYML_OPTIMIZER_FACTORY_HPP
#define HAPPYML_OPTIMIZER_FACTORY_HPP

#include "enums.hpp"
#include "mbgd_optimizer.hpp"
#include "adam_optimizer.hpp"
#include "sgdm_optimizer.hpp"

using namespace happyml;

namespace happyml {

    shared_ptr<BaseOptimizer> createOptimizer(const OptimizerType optimizerType,
                                              const float learningRate,
                                              const float biasLearningRate) {
        shared_ptr<BaseOptimizer> result;
        switch (optimizerType) {
            case OptimizerType::microbatch:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
                break;
            case OptimizerType::adam:
                result = make_shared<AdamOptimizer>(learningRate, biasLearningRate);
                break;
            case OptimizerType::sgdm:
                result = make_shared<SGDMOptimizer>(learningRate, biasLearningRate);
                break;
            default:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
        }
        return result;
    }
}
#endif // HAPPYML_OPTIMIZER_FACTORY_HPP