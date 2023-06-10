//
// Created by Erik Hyrkas on 6/7/2023.
// Copyright 2023. Usable under MIT license.
//


#include <memory>
#include "../ml/hyperband/hyperband.hpp"
#include "../util/dataset_utils.hpp"
#include "../lang/execution_context.hpp"

using namespace std;
using namespace happyml;

int main() {
    try {
        string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
        string result_path = base_path + "titanic/dataset.bin";
        auto titanicDataSource = make_shared<BinaryDataSet>(result_path, 0.9);
        auto titanicTestDatasource = make_shared<BinaryDataSet>(result_path, -0.1);

        std::shared_ptr<HyperparameterSpace> hyperparameterSpace = std::make_shared<HyperparameterSpace>(titanicDataSource->recordCount() / 10);
        std::shared_ptr<ConfigurationEvaluator> configurationEvaluator =
                std::make_shared<ConfigurationEvaluator>("accuracy",
                                                         LossType::categoricalCrossEntropy,
                                                         OptimizerType::adam,
                                                         result_path, 0.9,
                                                         result_path, -0.1,
                                                         DEFAULT_HAPPYML_REPO_PATH,
                                                         5, MINUTE_MS);

        int maxResources = 1000;
        int reductionFactor = 3;

        Hyperband hyperband(hyperparameterSpace, configurationEvaluator, maxResources, reductionFactor);
        auto best_model = hyperband.run(30, 0.95f);
        std::shared_ptr<Hyperparameters> bestConfiguration = configurationEvaluator->getBestConfiguration();
        float bestEvaluationMetric = configurationEvaluator->getBestEvaluationMetric();
        std::cout << std::fixed << std::setprecision(3) << "Best Result: " << bestEvaluationMetric << " Best Configuration: " << bestConfiguration->as_string() << std::endl;
        best_model->setSilentMode(true);
        float loss = best_model->test(titanicTestDatasource);
        std::cout << std::fixed << std::setprecision(3) << "Test Loss: " << loss << std::endl;
        float accuracy = best_model->compute_categorical_accuracy(titanicTestDatasource, configurationEvaluator->getExpectedDecoders());
        std::cout << "Test Accuracy: " << accuracy << std::endl;
    } catch (std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}