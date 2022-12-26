//
// Created by Erik Hyrkas on 12/26/2022.
//
#ifndef MICROML_OPTIMIZER_FACTORY_HPP
#define MICROML_OPTIMIZER_FACTORY_HPP

#include "enums.hpp"
#include "mbgd_optimizer.hpp"

using namespace microml;

namespace microml {

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
#endif // MICROML_OPTIMIZER_FACTORY_HPP