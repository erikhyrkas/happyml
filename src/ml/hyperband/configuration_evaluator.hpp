//
// Created by Erik Hyrkas on 6/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CONFIGURATION_EVALUATOR_HPP
#define HAPPYML_CONFIGURATION_EVALUATOR_HPP

#include <utility>

#include "hyperband_space.hpp"
#include "../neural_network.hpp"
#include "../../util/encoder_decoder_builder.hpp"
#include "../happyml_dsl.hpp"

namespace happyml {
    class ConfigurationEvaluator {
    public:
        ConfigurationEvaluator(LossType loss_type, OptimizerType optimizer_type,
                               string dataset_path, float dataset_split,
                               string test_dataset_path, float test_dataset_split,
                               string repo_base_path, int max_epochs, int64_t max_time) :
                loss_type_(loss_type), optimizer_type_(optimizer_type),
                dataset_path_(std::move(dataset_path)), dataset_split_(dataset_split),
                test_dataset_path_(std::move(test_dataset_path)), test_dataset_split_(test_dataset_split),
                repo_base_path_(std::move(repo_base_path)), max_epochs_(max_epochs), max_time_(max_time) {
            best_configuration_ = nullptr;
            best_evaluation_metric_ = -INFINITY;
            best_model_name_ = "";
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

            std::stringstream ss;
            ss << std::hex << repo_base_path_ << "cache/" << "temp_" << timestamp;
            best_model_repo_ = ss.str();
            shared_ptr<BinaryDataSet> temp = make_shared<BinaryDataSet>(dataset_path_, dataset_split_);
            expected_decoders_ = build_expected_decoders(false, temp);
            minimize_metric_ = loss_type_ != LossType::categoricalCrossEntropy && loss_type_ != LossType::binaryCrossEntropy;
        }

        void remove_temp_folder() {
            // if exists
            if (std::filesystem::exists(best_model_repo_)) {
                std::filesystem::remove_all(best_model_repo_);
            }
        }

        bool evaluateConfiguration(const shared_ptr<Hyperparameters> &configuration, int allocatedResources, float targetMetric) {
            shared_ptr<NeuralNetworkForTraining> model = buildModel(configuration);
            shared_ptr<TrainingResult> trainingResult = trainModel(model, configuration, allocatedResources, max_epochs_, max_time_);
            float evaluationMetric = evaluateModel(model, trainingResult);
            {
                std::lock_guard<std::mutex> guard(mutex_);
                storeResults(configuration, evaluationMetric);
                if (updateBestConfiguration(model->get_name(), configuration, evaluationMetric)) {
                    return (minimize_metric_ && best_evaluation_metric_ < targetMetric) || (!minimize_metric_ && best_evaluation_metric_ > targetMetric);
                }
            }
            return false;
        }

        pair<shared_ptr<NeuralNetworkForTraining>, shared_ptr<TrainingResult>> buildAndTrain(const shared_ptr<Hyperparameters> &configuration) {
            shared_ptr<NeuralNetworkForTraining> model = buildModel(configuration);
            shared_ptr<TrainingResult> trainingResult = trainModel(model, configuration);
            return {model, trainingResult};
        }

        std::shared_ptr<Hyperparameters> getBestConfiguration() {
            return best_configuration_;
        }

        [[nodiscard]] float getBestEvaluationMetric() const {
            return best_evaluation_metric_;
        }

        [[nodiscard]] bool getMinimizeMetric() const {
            return minimize_metric_;
        }

        shared_ptr<NeuralNetworkForTraining> getBestModel() {
            auto loadedNeuralNetwork = loadNeuralNetworkForTraining(best_model_name_,
                                                                    best_model_repo_);
            return loadedNeuralNetwork;
        }

        std::vector<shared_ptr<RawDecoder>> getExpectedDecoders() {
            return expected_decoders_;
        }

    private:
        std::mutex mutex_;
        LossType loss_type_;
        OptimizerType optimizer_type_;
        string dataset_path_;
        float dataset_split_;
        string test_dataset_path_;
        float test_dataset_split_;
        std::vector<shared_ptr<RawDecoder>> expected_decoders_;
        shared_ptr<Hyperparameters> best_configuration_;
        float best_evaluation_metric_;
        bool minimize_metric_;
        std::string best_model_name_;
        std::string best_model_repo_;
        std::vector<pair<shared_ptr<Hyperparameters>, float>> results_;
        std::string repo_base_path_;
        int max_epochs_;
        int64_t max_time_;

        shared_ptr<NeuralNetworkForTraining> buildModel(const shared_ptr<Hyperparameters> &configuration) {
            auto neural_network_builder = neuralNetworkBuilder(optimizer_type_);
            auto dsl = neural_network_builder
                    ->setModelName(configuration->temp_folder_name())
                    ->setModelRepo(best_model_repo_)
                    ->setLearningRate(configuration->learning_rate)
                    ->setBiasLearningRate(configuration->bias_learning_rate)
                    ->setLossFunction(loss_type_);
            shared_ptr<BinaryDataSet> data_set = make_shared<BinaryDataSet>(dataset_path_, dataset_split_);
            auto givens = data_set->getGivenMetadata();
            auto givens_size = givens.size();
            // if there is only one input:
            //    * if it is an image, we can use a convolutional network.
            //    * else we can use a dense network.
            // if there are multiple inputs, we can concatenate them and use a dense network.
            size_t total_given_values = 0;
            for (auto &given_shape: data_set->getGivenShapes()) {
                if (given_shape.size() == 3) {
                    total_given_values += given_shape[0] * given_shape[1] * given_shape[2];
                } else if (given_shape.size() == 2) {
                    total_given_values += given_shape[0] * given_shape[1];
                } else if (given_shape.size() == 1) {
                    total_given_values += given_shape[0];
                } else {
                    throw runtime_error("Unsupported given shape size");
                }
            }
            auto desired_width = (size_t) ((float) total_given_values * configuration->complexity_width);
            auto number_of_filters = (size_t) configuration->complexity_width;
            size_t total_expected_values = 0;
            for (auto &expected_shape: data_set->getExpectedShapes()) {
                if (expected_shape.size() == 3) {
                    total_expected_values += expected_shape[0] * expected_shape[1] * expected_shape[2];
                } else if (expected_shape.size() == 2) {
                    total_expected_values += expected_shape[0] * expected_shape[1];
                } else if (expected_shape.size() == 1) {
                    total_expected_values += expected_shape[0];
                } else {
                    throw runtime_error("Unsupported expected shape size");
                }
            }
            auto output_size_expected = (size_t) ((float) total_expected_values * configuration->complexity_width);
            bool use_cnn = false;
            shared_ptr<HappymlDSL::NNVertex> last_layer;
            if (givens_size == 1) {
                auto given = givens[0];
                if (given->purpose == 'I') {
                    use_cnn = true;
                }
                if (use_cnn) {
                    last_layer = dsl->addInputLayer(data_set->getGivenShape(), number_of_filters, 3, LayerType::convolution2dValid,
                                                    ActivationType::relu);
                } else {
                    last_layer = dsl->addInputLayer(data_set->getGivenShape(),
                                                    desired_width,
                                                    LayerType::full,
                                                    ActivationType::relu);
                }
            } else {
                last_layer = dsl->add_concatenated_input_layer(data_set->getGivenShapes());
                last_layer = last_layer->addLayer(desired_width, LayerType::full,
                                                  ActivationType::relu);
            }
            apply_layer_settings(last_layer, configuration, true);
            float diminishing_dropout_rate = configuration->dropout_rate;
            if (diminishing_dropout_rate > 1e-8) {
                last_layer = last_layer->addDropoutLayer(configuration->dropout_rate);
                diminishing_dropout_rate *= 0.2f;
            }

            for (int i = 0; i < configuration->complexity_depth; i++) {
                // add hidden layers
                if (use_cnn) {
                    last_layer = last_layer->addLayer(number_of_filters, 3, LayerType::convolution2dValid, ActivationType::relu);
                } else {
                    last_layer = last_layer->addLayer(desired_width, LayerType::full,
                                                      ActivationType::relu);
                }
                apply_layer_settings(last_layer, configuration, false);
                if (diminishing_dropout_rate >= 0.2) {
                    last_layer = last_layer->addDropoutLayer(configuration->dropout_rate);
                    diminishing_dropout_rate *= 0.2f;
                }
            }

            ActivationType last_activation;
            if (loss_type_ == LossType::categoricalCrossEntropy) {
                last_activation = ActivationType::softmax;
                // TODO: this is a hack to improve the accuracy of softmax. this really shouldn't need to be done.
                last_layer = last_layer->addLayer(output_size_expected, LayerType::full, ActivationType::sigmoid);
            } else {
                last_activation = ActivationType::sigmoid;
            }
            for (int i = 0; i < data_set->getExpectedShapes().size(); i++) {
                auto next_expected_shape = data_set->getExpectedShapes()[i];
                auto output_layer = last_layer
                        ->addOutputLayer(next_expected_shape,
                                         last_activation)
                        ->setUseBias(configuration->use_bias);
            }
            auto neuralNetwork = neural_network_builder->build();
            return neuralNetwork;
        }

        shared_ptr<TrainingResult> trainModel(shared_ptr<NeuralNetworkForTraining> &model,
                                              const shared_ptr<Hyperparameters> &configuration,
                                              int allocatedResources = -1,
                                              int max_epochs = 1000000,
                                              int64_t max_time = NINETY_DAYS_MS) {
            int patience = allocatedResources < 1 ? 20 : allocatedResources;
            float improvement_tolerance = 1e-7;
            auto exit_strategy = make_shared<DefaultExitStrategy>(patience,
                                                                  max_time,
                                                                  max_epochs,
                                                                  1e-3,
                                                                  improvement_tolerance,
                                                                  2,
                                                                  0.25f);
            model->setSilentMode(true);
            shared_ptr<BinaryDataSet> data_set = make_shared<BinaryDataSet>(dataset_path_, dataset_split_);
            shared_ptr<BinaryDataSet> test_data_set = make_shared<BinaryDataSet>(test_dataset_path_, test_dataset_split_);
            auto trainingResults = model->train(data_set, test_data_set, configuration->batch_size);
            model->saveWithOverwrite();
            return trainingResults;
        }

        float evaluateModel(const shared_ptr<NeuralNetworkForTraining> &model, const shared_ptr<TrainingResult> &trainingResult) {
            shared_ptr<BinaryDataSet> test_data_set = make_shared<BinaryDataSet>(test_dataset_path_, test_dataset_split_);
            if (loss_type_ == LossType::categoricalCrossEntropy) {
                return model->compute_categorical_accuracy(test_data_set, expected_decoders_);
            } else if (loss_type_ == LossType::binaryCrossEntropy) {
                return model->compute_binary_accuracy(test_data_set);
            }
            return trainingResult->final_test_loss;
        }

        void storeResults(const shared_ptr<Hyperparameters> &configuration, float evaluationMetric) {
            pair<shared_ptr<Hyperparameters>, float> result(configuration, evaluationMetric);
            results_.push_back(result);
        }

        bool updateBestConfiguration(std::string model_name, const shared_ptr<Hyperparameters> &configuration, float evaluationMetric) {
            configuration->evaluation_metric = evaluationMetric;
            configuration->minimize_metric = minimize_metric_;
            if (best_configuration_ == nullptr ||
                (minimize_metric_ && evaluationMetric < best_evaluation_metric_) ||
                (!minimize_metric_ && evaluationMetric > best_evaluation_metric_)) {
                best_configuration_ = configuration;
                best_evaluation_metric_ = evaluationMetric;
                best_model_name_ = std::move(model_name);
                return true;
            }
            return false;
        }

        static void apply_layer_settings(const shared_ptr<HappymlDSL::NNVertex> &last_layer, const shared_ptr<Hyperparameters> &configuration, bool force_32_bits) {
            if (!force_32_bits && configuration->bits != 32) {
                last_layer->setBits(configuration->bits)->setMaterialized(false);
            }
            last_layer->setUseBias(configuration->use_hidden_bias);
            if (configuration->l2_regularization_strength > 1e-7) {
                last_layer->setUseL2Regularization(true);
                last_layer->setRegularizationStrength(configuration->l2_regularization_strength);
            } else {
                last_layer->setUseL2Regularization(false);
            }
            last_layer->setUseNormClipping(configuration->use_normal_clipping);
            last_layer->setUseNormalization(configuration->use_normalization_layers);
        }
    };

}
#endif //HAPPYML_CONFIGURATION_EVALUATOR_HPP
