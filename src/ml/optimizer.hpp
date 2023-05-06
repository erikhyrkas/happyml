//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_OPTIMIZER_HPP
#define HAPPYML_OPTIMIZER_HPP

#include "../types/base_tensors.hpp"

namespace happyml {
    class BaseOptimizer {
    public:
        // This framework tracks the learning rate of weights and bias
        // separately. This has to do with the underlying implementation of
        // quarterfloats and how loss of precision can be impacted
        explicit BaseOptimizer(float learningRate, float biasLearningRate)
                : learningRate(learningRate), biasLearningRate(biasLearningRate) {

        }

        // Neurons (aka layers) must register with the optimizer before training.
        // This allows the optimizer to initialize state it will need to track that
        // specific neuron's state. This is particularly useful for optimizers
        // that calculate momentum.

        // This method is for neurons (layers) that have weight changes. The id should be
        // unique and not be reused for bias, since not all neurons (layers)
        // have a bias. This method will initialize any state we'll need to track
        // for that specific layer and that layer will then give us back this same id in
        // the calculateWeightsChange call.
        virtual int registerForWeightChanges() {
            return 0;
        }

        // This method is for neurons (layers) that have bias changes. The id should be
        // unique and not be reused for weights, since not all neurons (layers)
        // have a bias. This method will initialize any state we'll need to track
        // for that specific layer and that layer will then give us back this same id in
        // the calculateBiasChange call.
        virtual int registerForBiasChanges() {
            return 0;
        }

        // We only calculate changes to weight in this method. If changes to bias need to be calculated
        // it will be done by the neuron calling calculateBiasChange. The layer
        // needs to have a valid registration id for weights, which it can get by calling registerForWeightChanges().
        // The registration_id allows us to use any state we need to track for that specific layer.
        // useful in optimizers that calculate momentum.
        virtual shared_ptr<BaseTensor> calculateWeightsChange(int registration_id,
                                                              const shared_ptr<BaseTensor> &weights,
                                                              const shared_ptr<BaseTensor> &loss_gradient) = 0;

        // We only calculate changes to bias in this method. If changes to weight need to be calculated
        // it will be done by the neuron calling calculateWeightsChange. The layer
        // needs to have a valid registration id for bias, which it can get by calling registerForBiasChanges().
        // The registration_id allows us to use any state we need to track for that specific layer.
        // useful in optimizers that calculate momentum.
        virtual shared_ptr<BaseTensor> calculateBiasChange(int registration_id,
                                                           const shared_ptr<BaseTensor> &bias,
                                                           const shared_ptr<BaseTensor> &loss_gradient) = 0;

        // allows the caller to inspect the learning rate for weights
        [[nodiscard]] float getLearningRate() const {
            return learningRate;
        }

        // allows the caller to inspect the learning rate for bias
        [[nodiscard]] float getBiasLearningRate() const {
            return biasLearningRate;
        }

        void update_time_step() {
            time_step++;
        }

    protected:
        float learningRate;
        float biasLearningRate;
        int time_step = 0;
    };

}
#endif //HAPPYML_OPTIMIZER_HPP
