//
// Created by Erik Hyrkas on 6/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HYPERBAND_RANDOM_SEARCH_HPP
#define HAPPYML_HYPERBAND_RANDOM_SEARCH_HPP

#include <random>
#include "hyperband_space.hpp"

namespace happyml {
    class HyperBandRandomSearch {
    public:
        explicit HyperBandRandomSearch(const std::shared_ptr<HyperparameterSpace> &hyperparameterSpace)
                : hyperparameterSpace(hyperparameterSpace), randomEngine(std::random_device{}()) {
        }

        std::shared_ptr<Hyperparameters> generateRandomConfiguration(int bits_per_hyperparameter) {
            auto configuration = internal_random_config(bits_per_hyperparameter);
            std::string configurationString = configuration->as_string();
            // while improbable, let's just make sure we don't generate the same configuration twice
            int maxAttempts = 1000;
            while (usedConfigurations.find(configurationString) != usedConfigurations.end()) {
                configuration = internal_random_config(bits_per_hyperparameter);
                configurationString = configuration->as_string();
                maxAttempts--;
                if (maxAttempts == 0) {
                    // against all odds, we'll return a duplicate configuration if we've tried too many times
                    break;
                }
            }
            usedConfigurations.insert(configuration->as_string());
            return configuration;
        }

    private:
        // Helper function to get a random value from a range or set of values
        template<typename T>
        T getRandomValue(const std::vector<T> &values) {
            std::uniform_int_distribution<int> distribution(0, values.size() - 1);
            int index = distribution(randomEngine);
            return values[index];
        }

        // Instead of completely random values, weight the values by their respective weights
        // a higher weight means a higher probability of being chosen
        template<typename T>
        T getWeightedRandomValue(const std::vector<T> &values, float distributionFavor) {
            std::vector<float> weights = buildWeightsVector(values.size(), distributionFavor);
            std::discrete_distribution<int> distribution(weights.begin(), weights.end());
            int index = distribution(randomEngine);
            return values[index];
        }

        // a distribution favor above 1 favors later values
        // a distribution favor of 1 is a uniform distribution
        // a distribution favor of less than one favors earlier values
        // a distribution favor of 0 is a distribution that only chooses the first value
        static std::vector<float> buildWeightsVector(int numValues, float distributionFavor) {
            std::vector<float> weights(numValues);

            for (int i = 0; i < numValues; i++) {
                weights[i] = std::pow(distributionFavor, (float) i);
            }

            // Normalize the weights to sum up to 1
            float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
            for (float &weight: weights) {
                weight /= sum;
            }

            return weights;
        }

        shared_ptr<Hyperparameters> internal_random_config(int bitsPerHyperparameter) {
            auto configuration = make_shared<Hyperparameters>();
            configuration->learning_rate = getWeightedRandomValue(hyperparameterSpace->learning_rate_space, 0.9);
            configuration->bias_learning_rate = getWeightedRandomValue(hyperparameterSpace->bias_learning_rate_space, 0.9);
            configuration->complexity_depth = getWeightedRandomValue(hyperparameterSpace->complexity_depth_space, 0.7);
            configuration->complexity_width = getWeightedRandomValue(hyperparameterSpace->complexity_width_space, 0.7);
            configuration->dropout_rate = getWeightedRandomValue(hyperparameterSpace->dropout_rate_space, 0.5);
            configuration->l2_regularization_strength = getWeightedRandomValue(hyperparameterSpace->l2_regularization_strength_space, 0.5);
            configuration->batch_size = getWeightedRandomValue(hyperparameterSpace->batch_size_space, 0.5);
            configuration->use_normalization_layers = getWeightedRandomValue(hyperparameterSpace->use_normalization_layers_space, 0.1);
            configuration->use_hidden_bias = getWeightedRandomValue(hyperparameterSpace->use_hidden_bias_space, 0.1);
            configuration->use_bias = getWeightedRandomValue(hyperparameterSpace->use_bias_space, 0.9);
            configuration->use_normal_clipping = getWeightedRandomValue(hyperparameterSpace->use_normal_clipping_space, 0.3);
            configuration->bits = bitsPerHyperparameter;
            return configuration;
        }

        std::unordered_set<std::string> usedConfigurations;
        // Random engine for generating random numbers
        std::default_random_engine randomEngine;

        // Hyperparameter space
        const std::shared_ptr<HyperparameterSpace> hyperparameterSpace;
    };

}
#endif //HAPPYML_HYPERBAND_RANDOM_SEARCH_HPP
