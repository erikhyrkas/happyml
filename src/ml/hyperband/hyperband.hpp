//
// Created by Erik Hyrkas on 6/9/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HYPERBAND_HPP
#define HAPPYML_HYPERBAND_HPP

#include "hyperband_space.hpp"
#include "resource_allocator.hpp"
#include "configuration_evaluator.hpp"
#include "hyperband_random_search.hpp"

namespace happyml {
    class Hyperband {
    public:
        Hyperband(shared_ptr<HyperparameterSpace> &hyperparameter_space, shared_ptr<ConfigurationEvaluator> &configuration_evaluator, int max_resources, int reduction_factor)
                : hyperparameter_space_(hyperparameter_space), configuration_evaluator_(configuration_evaluator), max_resources(max_resources), reduction_factor(reduction_factor) {
        }

        shared_ptr<NeuralNetworkForTraining> run(int num_configurations = -1, float target_metric = 0.95f) {
            try {
                if (num_configurations == -1 || num_configurations > hyperparameter_space_->getNumConfigurations()) {
                    num_configurations = std::min(hyperparameter_space_->getNumConfigurations(), std::max(10, hyperparameter_space_->getNumConfigurations() / 10000));
                }
                std::vector<shared_ptr<Hyperparameters>> configurations = generateInitialConfigurations(num_configurations);

                for (int round = 0; num_configurations > 1; round++) {
                    int allocated_resources = std::max(2, max_resources / static_cast<int>(pow(reduction_factor, round)));

                    std::cout << "Starting round " << round << " with " << configurations.size() << " configurations." << endl;
                    std::for_each(std::execution::par, configurations.begin(), configurations.end(), [&](shared_ptr<Hyperparameters> &configuration) {
                        cout << "{";
                        configuration_evaluator_->evaluateConfiguration(configuration, allocated_resources, target_metric);
                        cout << "}";
                    });
                    cout << endl;
                    float best_evaluation_metric = configuration_evaluator_->getBestEvaluationMetric();
                    cout << endl << "Best evaluation metric: " << best_evaluation_metric << endl;
                    if ((configuration_evaluator_->getMinimizeMetric() && best_evaluation_metric < target_metric) || (!configuration_evaluator_->getMinimizeMetric() && best_evaluation_metric > target_metric)) {
                        cout << "Stopping early, found a very good candidate." << endl;
                        break;
                    }

//                    bool done = false;
//                    for (shared_ptr<Hyperparameters> &configuration: configurations) {
//                        if (configuration_evaluator_->evaluateConfiguration(configuration, allocated_resources, target_metric)) {
//                            cout << "Stopping early, found a very good candidate." << endl;
//                            done = true;
//                            break;
//                        }
//                    }
//                    if (done) {
//                        break;
//                    }

                    eliminateConfigurations(configurations, num_configurations);

                    num_configurations /= reduction_factor;
                    configurations.resize(num_configurations);
                }
                auto best_model = configuration_evaluator_->getBestModel();
                configuration_evaluator_->remove_temp_folder();
                return best_model;
            } catch (std::exception &e) {
                cout << "Exception: " << e.what() << endl;
                configuration_evaluator_->remove_temp_folder();
            }
            return nullptr;
        }

    private:
        std::vector<shared_ptr<Hyperparameters>> generateInitialConfigurations(int num_configurations) {
            std::vector<shared_ptr<Hyperparameters>> configurations(num_configurations);

            HyperBandRandomSearch randomSearch(hyperparameter_space_);

            for (int i = 0; i < num_configurations; i++) {
                configurations[i] = randomSearch.generateRandomConfiguration();
            }

            return configurations;
        }

        void eliminateConfigurations(std::vector<shared_ptr<Hyperparameters>> &configurations, int &num_configurations) const {
            std::partial_sort(configurations.begin(), configurations.begin() + num_configurations / reduction_factor, configurations.end(),
                              [](const shared_ptr<Hyperparameters> &config1, const shared_ptr<Hyperparameters> &config2) {
                                  // return true if config1 is worse than config2
                                  return (config1->minimize_metric && config1->evaluation_metric > config2->evaluation_metric) ||
                                         (!config1->minimize_metric && config1->evaluation_metric < config2->evaluation_metric);
                              });

            configurations.erase(configurations.begin() + num_configurations / reduction_factor, configurations.end());
        }

        shared_ptr<HyperparameterSpace> hyperparameter_space_;
        int max_resources;
        int reduction_factor;
        shared_ptr<ConfigurationEvaluator> configuration_evaluator_;
    };
}
#endif //HAPPYML_HYPERBAND_HPP
