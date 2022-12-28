//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//
#ifndef HAPPYML_OPTIMIZER_FACTORY_HPP
#define HAPPYML_OPTIMIZER_FACTORY_HPP

#include "enums.hpp"
#include "mbgd_optimizer.hpp"

using namespace happyml;

namespace happyml {

    shared_ptr<Optimizer> createOptimizer(const OptimizerType optimizerType,
                                          const float learningRate,
                                          const float biasLearningRate) {
        shared_ptr<Optimizer> result;
        switch (optimizerType) {
            case OptimizerType::microbatch:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
                break;
            default:
                result = make_shared<MBGDOptimizer>(learningRate, biasLearningRate);
        }
        return result;
    }
}
#endif // HAPPYML_OPTIMIZER_FACTORY_HPP