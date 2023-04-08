//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_OPTIMIZER_HPP
#define HAPPYML_OPTIMIZER_HPP

#include "../types/tensor.hpp"

namespace happyml {
    class BaseOptimizer {
    public:
        explicit BaseOptimizer(float learningRate, float biasLearningRate)
                : learningRate(learningRate), biasLearningRate(biasLearningRate) {}


        virtual int registerForWeightChanges() = 0;

        virtual int registerForBiasChanges() = 0;

        virtual shared_ptr<BaseTensor> calculateWeightsChange(int registration_id,
                                                              const shared_ptr<BaseTensor> &weights,
                                                              const shared_ptr<BaseTensor> &loss_gradient,
                                                              float mixedPrecisionScale) = 0;

        virtual shared_ptr<BaseTensor> calculateBiasChange(int registration_id,
                                                           const shared_ptr<BaseTensor> &bias,
                                                           const shared_ptr<BaseTensor> &loss_gradient,
                                                           float mixedPrecisionScale,
                                                           float current_batch_size) = 0;

        [[nodiscard]] float getLearningRate() const {
            return learningRate;
        }

        [[nodiscard]] float getBiasLearningRate() const {
            return biasLearningRate;
        }

    protected:
        float learningRate;
        float biasLearningRate;
    };

}
#endif //HAPPYML_OPTIMIZER_HPP
