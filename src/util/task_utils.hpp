//
// Created by Erik Hyrkas on 5/14/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TASK_UTILS_HPP
#define HAPPYML_TASK_UTILS_HPP

#include "dataset_utils.hpp"
#include "../ml/happyml_dsl.hpp"
#include "../../util/pretty_print_row.hpp"
#include "../lang/happyml_variant.hpp"

namespace happyml {

    size_t estimate_layer_output_size(const vector<vector<size_t>> &given_shapes,
                                      const vector<vector<size_t>> &expected_shapes,
                                      const string &goal,
                                      double complexity_multiplier) {
        size_t total_values = 0;
        for (auto &given_shape: given_shapes) {
            if (given_shape.size() == 3) {
                total_values += given_shape[0] * given_shape[1] * given_shape[2];
            } else if (given_shape.size() == 2) {
                total_values += given_shape[0] * given_shape[1];
            } else if (given_shape.size() == 1) {
                total_values += given_shape[0];
            } else {
                throw runtime_error("Unsupported given shape size");
            }
        }
        size_t total_output_values = 0;
        for (auto &expected_shape: expected_shapes) {
            if (expected_shape.size() == 3) {
                total_output_values += expected_shape[0] * expected_shape[1] * expected_shape[2];
            } else if (expected_shape.size() == 2) {
                total_output_values += expected_shape[0] * expected_shape[1];
            } else if (expected_shape.size() == 1) {
                total_output_values += expected_shape[0];
            } else {
                throw runtime_error("Unsupported expected shape size");
            }
        }

        auto max_val = (size_t) ((double) total_output_values * (0.25 + complexity_multiplier));
        total_values = std::min(total_values, max_val);
        return total_values;
    }

    size_t estimate_layer_output_size(vector<vector<size_t>> expected_shapes, const string &goal, double complexity_multiplier) {
        size_t total_values = 0;
        for (auto &expected_shape: expected_shapes) {
            if (expected_shape.size() == 3) {
                total_values += expected_shape[0] * expected_shape[1] * expected_shape[2];
            } else if (expected_shape.size() == 2) {
                total_values += expected_shape[0] * expected_shape[1];
            } else if (expected_shape.size() == 1) {
                total_values += expected_shape[0];
            } else {
                throw runtime_error("Unsupported expected shape size");
            }
        }
        return (size_t) ((double) total_values * complexity_multiplier);
    }

    vector<shared_ptr<BaseTensor>> build_given(const unordered_map<std::string, std::vector<HappyMLVariant>> &user_inputs, const vector<shared_ptr<HappyMLVariantEncoder>> &encoders) {
        vector<shared_ptr<BaseTensor>> given;
        for (auto &encoder: encoders) {
            string column_name = encoder->get_name();
            // the column name case needs to be insensitive, as we do that by always being lower case.
            std::transform(column_name.begin(), column_name.end(), column_name.begin(), ::tolower);
            // skip if it doesn't exist
            if (user_inputs.find(column_name) == user_inputs.end()) {
                break;
            }
            const vector<HappyMLVariant> &user_input = user_inputs.at(column_name);
            try {
                auto given_tensor = encoder->encode(user_input);
                given.push_back(given_tensor);
            } catch (const std::exception &e) {
                cout << "Error while encoding input: " << e.what() << endl;
                return given;
            }
        }
        return given;
    }

    bool execute_task_with_inputs(const string &task_name, const std::unordered_map<std::string, std::vector<HappyMLVariant>> &inputs, const string &task_folder_path) {
        string task_full_path = task_folder_path + task_name;
        if (!filesystem::exists(task_full_path)) {
            cout << "Task " << task_name << " does not exists. Skipping." << endl;
            return false;
        }
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining(task_name,
                                                                task_folder_path);

        string dataset_full_file_path = task_full_path + "/dataset.bin";
        BinaryDatasetReader reader(dataset_full_file_path);
        vector<shared_ptr<RawDecoder>> given_decoders = build_given_decoders(false, reader);
        vector<shared_ptr<RawDecoder>> expected_decoders = build_expected_decoders(false, reader);
        vector<shared_ptr<HappyMLVariantEncoder>> given_encoders = build_given_encoders(reader);

        vector<string> given_column_names = reader.get_given_names();
        vector<string> expected_column_names = reader.get_expected_names();
        reader.close();

        vector<size_t> widths;

        cout << "Results: " << endl;
        auto given_values = build_given(inputs, given_encoders);
        if (given_values.size() != given_column_names.size()) {
            stringstream message;
            // the keys of inputs didn't match given_column_names
            // let's print the values we expected from given_column_names that weren't in inputs and the values of keys in inputs that weren't in given_column_names
            // to provide a good error message.
            bool found_one = false;
            string delim = "You did not provide the fields: ";
            for (auto &given_column_name: given_column_names) {
                string lower_given_column_name = given_column_name;
                std::transform(lower_given_column_name.begin(), lower_given_column_name.end(), lower_given_column_name.begin(), ::tolower);
                if (inputs.find(lower_given_column_name) == inputs.end()) {
                    message << delim << given_column_name;
                    delim = ",";
                    found_one = true;
                }
            }
            if (found_one) {
                message << ". ";
            }
            vector<string> lower_given_column_names;
            for (auto &given_column_name: given_column_names) {
                string lower_given_column_name = given_column_name;
                std::transform(lower_given_column_name.begin(), lower_given_column_name.end(), lower_given_column_name.begin(), ::tolower);
                lower_given_column_names.push_back(lower_given_column_name);
            }
            delim = "You provided the invalid fields: ";
            found_one = false;
            for (auto &input: inputs) {
                string lower_input_name = input.first;
                std::transform(lower_input_name.begin(), lower_input_name.end(), lower_input_name.begin(), ::tolower);
                if (std::find(lower_given_column_names.begin(), lower_given_column_names.end(), lower_input_name) == lower_given_column_names.end()) {
                    message << delim << input.first;
                    delim = ",";
                    found_one = true;
                }
            }
            if (found_one) {
                message << ".";
            }
            cout << message.str() << endl;
            return false;
        }
        auto predictions = loadedNeuralNetwork->predict(given_values);

        auto expected_values = record_group_to_strings(expected_decoders, predictions);

        if (widths.empty()) {
            // using width of first result is suboptimal, but it's good enough for now.
            widths = calculate_pretty_print_column_widths(expected_column_names, expected_values);
            pretty_print_header(cout, expected_column_names, widths);
        }
        pretty_print_row(cout, expected_values, widths);
        return true;
    }

    bool execute_task_with_dataset(const string &task_name, const string &dataset_file_path, const string &task_folder_path) {
        // TODO: we need to cache the task in the context so we don't have to reload it.

        string task_full_path = task_folder_path + task_name;
        if (!filesystem::exists(task_full_path)) {
            cout << "Task " << task_name << " does not exists. Skipping." << endl;
            return false;
        }

        // TODO: this doesn't use label, yet. It assumes "default" label.
        // TODO: this could use NeuralNetworkForPrediction instead of NeuralNetworkForTraining
        //  while NeuralNetworkForTraining works, it uses more memory and is slower.
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining(task_name,
                                                                task_folder_path);

        string dataset_full_file_path = dataset_file_path + "/dataset.bin";
        BinaryDatasetReader reader(dataset_full_file_path);
        vector<shared_ptr<RawDecoder>> given_decoders = build_given_decoders(false, reader);
        vector<shared_ptr<RawDecoder>> expected_decoders = build_expected_decoders(false, reader);
        vector<string> given_column_names = reader.get_given_names();
        vector<string> expected_column_names = reader.get_expected_names();
        auto given_metadata = reader.get_given_metadata();
        auto expected_metadata = reader.get_expected_metadata();
        reader.close();

        vector<string> merged_headers = pretty_print_merge_headers(expected_column_names, given_column_names);
        vector<size_t> widths;
        cout << "Results: " << endl;
        auto dataset = make_shared<BinaryDataSet>(dataset_full_file_path, given_metadata, expected_metadata);
        auto nextRecord = dataset->nextRecord();
        while (nextRecord) {
            auto given_values = nextRecord->getGiven();
            auto predictions = loadedNeuralNetwork->predict(given_values);

            auto merged_values = pretty_print_merge_records(expected_decoders, predictions, given_decoders, given_values);
            if (widths.empty()) {
                // using width of first result is suboptimal, but it's good enough for now.
                widths = calculate_pretty_print_column_widths(merged_headers, merged_values);
                pretty_print_header(cout, merged_headers, widths);
            }
            pretty_print_row(cout, merged_values, widths);

            nextRecord = dataset->nextRecord();
        }

        return true;
    }

    shared_ptr<NeuralNetworkForTraining> build_neural_network_for_label(const string &task_name,
                                                                        const string &task_folder_path,
                                                                        const string &goal,
                                                                        shared_ptr<BinaryDataSet> &dataSource,
                                                                        BinaryDatasetReader &reader,
                                                                        float learningRateAdjustmentFactor,
                                                                        OptimizerType optimizerType,
                                                                        LossType lossType,
                                                                        ActivationType activationType,
                                                                        double complexity_modifier) {

        float learningRate;
        float biasLearningRate;
        switch (optimizerType) {
            case sgd:
                learningRate = 0.005f;
                biasLearningRate = 0.001f;
                break;
            default:
                learningRate = 0.001f;
                biasLearningRate = 0.001f;
        }

        const float decay_rate = 0.95f;
        learningRate *= std::powf(decay_rate, learningRateAdjustmentFactor);
        biasLearningRate *= std::powf(decay_rate, learningRateAdjustmentFactor);
        cout << std::fixed << std::setprecision(6) << "Using learning rate " << learningRate << " (" << biasLearningRate << "); ";

        auto neural_network_builder = neuralNetworkBuilder(optimizerType);
        auto initial_layers = neural_network_builder
                ->setModelName(task_name)
                ->setModelRepo(task_folder_path)
                ->setLearningRate(learningRate)
                ->setBiasLearningRate(biasLearningRate)
                ->setLossFunction(lossType);
        shared_ptr<HappymlDSL::NNVertex> layer1;
        bool is_convolutional = false;

        size_t output_size_given_expected = estimate_layer_output_size(dataSource->getGivenShapes(),
                                                                       dataSource->getExpectedShapes(),
                                                                       goal,
                                                                       complexity_modifier);
        if (dataSource->getGivenShapes().size() > 1) {
            shared_ptr<HappymlDSL::NNVertex> input_layer = initial_layers->add_concatenated_input_layer(dataSource->getGivenShapes());
            layer1 = input_layer
                    ->addLayer(output_size_given_expected,
                               LayerType::full, activationType);
            cout << "concatenated input layer (" << dataSource->getGivenShapes().size() << ") -> ";
        } else {
            is_convolutional = reader.get_given_metadata(0)->purpose == 'I';
            if (is_convolutional) {
                layer1 = initial_layers->addInputLayer(dataSource->getGivenShape(), 1, 3, LayerType::convolution2dValid,
                                                       activationType)->setUseBias(true);
                cout << "convolutional input layer (1 filter and kernel size 3) -> ";
            } else {
                layer1 = initial_layers->addInputLayer(dataSource->getGivenShape(),
                                                       output_size_given_expected,
                                                       LayerType::full, activationType)->setUseBias(true);
                cout << "full input layer: " << output_size_given_expected << " -> ";
            }
        }
        layer1->setUseL2Regularization(true);
        layer1->setUseNormClipping(true);
        if (lossType == LossType::categoricalCrossEntropy) {
            layer1->setUseNormalization(true);
            cout << "normalization layer -> ";
        }
        if (goal == "memory") {
            layer1 = layer1->setBits(8)->setMaterialized(false);
        }
        layer1 = layer1->addDropoutLayer(0.8f);

        shared_ptr<HappymlDSL::NNVertex> layer2;
        if (is_convolutional) {
            layer2 = layer1
                    ->addLayer(output_size_given_expected,
                               LayerType::full, activationType)->setUseBias(true);
            layer2->setUseL2Regularization(true);
            layer2->setUseNormClipping(true);
            cout << "full layer: " << output_size_given_expected << " -> ";
        } else {
            size_t output_size_expected = estimate_layer_output_size(dataSource->getExpectedShapes(), goal, complexity_modifier);
            layer2 = layer1
                    ->addLayer(output_size_expected,
                               LayerType::full, activationType)->setUseBias(true);
            layer2->setUseL2Regularization(true);
            layer2->setUseNormClipping(true);
            cout << "full second layer: " << output_size_expected << " -> ";
        }
        if (lossType == LossType::categoricalCrossEntropy) {
            layer2->setUseNormalization(true);
            cout << "normalization layer" << " -> ";
        }

        if (goal == "memory") {
            layer2 = layer2->setBits(8)->setMaterialized(false);
        }
        layer2 = layer2->addDropoutLayer(0.5f);

        ActivationType last_activation;
        if (lossType == LossType::categoricalCrossEntropy) {
            last_activation = ActivationType::softmax;
            // TODO: this is a hack to improve the accuracy of softmax. this really shouldn't need to be done.
            size_t output_size_expected = estimate_layer_output_size(dataSource->getExpectedShapes(), "speed", 1.0f);
            layer2 = layer2->addLayer(output_size_expected, LayerType::full, ActivationType::sigmoid);
        } else {
            last_activation = ActivationType::sigmoid;
        }
        for (int i = 0; i < dataSource->getExpectedShapes().size(); i++) {
            auto next_expected_shape = dataSource->getExpectedShapes()[i];
            auto output_layer = layer2
                    ->addOutputLayer(next_expected_shape,
                                     last_activation)
                    ->setUseBias(true);
            if (next_expected_shape[0] > 1 && next_expected_shape[2] > 1) {
                cout << "output layer: " << next_expected_shape[0] << ", " << next_expected_shape[1] << ", " << next_expected_shape[2] << endl;
            } else {
                cout << "output layer: " << next_expected_shape[1] << endl;
            }
        }
        auto neuralNetwork = neural_network_builder->build();
        return neuralNetwork;
    }

    string getAttemptKey(int batch_size, float learningRateAdjustmentFactor, double complexity_modifier) { return to_string(batch_size) + "_" + to_string(learningRateAdjustmentFactor) + "_" + to_string(complexity_modifier); }

    shared_ptr<TrainingResult> attempt_training_with_logging(const string &task_name, const string &goal, const string &task_folder_path, const shared_ptr<BinaryDataSet> &testDataSource, int max_batch_size, OptimizerType &optimizerType,
                                                             ActivationType &activationType, LossType &lossType, float learningRateSearchRate, size_t patience, size_t min_epochs, float improvement_tolerance,
                                                             const shared_ptr<DefaultExitStrategy> &exit_strategy,
                                                             vector<shared_ptr<RawDecoder>> &expected_decoders, float target_accuracy, int attempt, double complexity_modifier, float learningRateAdjustmentFactor, ofstream &log_file,
                                                             shared_ptr<BinaryDataSet> &dataSource, BinaryDatasetReader &reader, shared_ptr<NeuralNetworkForTraining> &neuralNetwork) {
        shared_ptr<TrainingResult> attempt_result;
        neuralNetwork = build_neural_network_for_label(task_name, task_folder_path, goal, dataSource, reader, learningRateAdjustmentFactor, optimizerType, lossType, activationType,
                                                       complexity_modifier);
        neuralNetwork->setExitStrategy(exit_strategy);
        try {
            if (testDataSource == nullptr) {
                attempt_result = neuralNetwork->train(dataSource, max_batch_size);
                attempt_result->accuracy = neuralNetwork->compute_categorical_accuracy(dataSource, expected_decoders);
            } else {
                attempt_result = neuralNetwork->train(dataSource, testDataSource, max_batch_size);
                attempt_result->accuracy = neuralNetwork->compute_categorical_accuracy(testDataSource, expected_decoders);
            }
            attempt_result->learningRateAdjustmentFactor = learningRateAdjustmentFactor;
            attempt_result->complexity_modifier = complexity_modifier;

            cout << "Accuracy: " << fixed << setprecision(2) << (attempt_result->accuracy * 100) << "%" << endl;
            log_file << "------------------" << attempt << "------------------" << endl;
            log_file << "\tTarget accuracy: " << target_accuracy << endl;
            log_file << "\tPatience: " << patience << endl;
            log_file << "\tMin epochs: " << min_epochs << endl;
            log_file << "\tImprovement tolerance: " << improvement_tolerance << endl;
            log_file << "\tLearning rate search rate: " << learningRateSearchRate << endl;
            log_file << "\tOptimizer: " << optimizerTypeToString(optimizerType) << endl;
            log_file << "\tLoss: " << lossTypeToString(lossType) << endl;
            log_file << "\tActivation: " << activationTypeToString(activationType) << endl;
            log_file << "\tInitial test loss: " << fixed << setprecision(4) << attempt_result->initial_test_loss << endl;
            log_file << "\tFinal test loss: " << fixed << setprecision(4) << attempt_result->final_test_loss << endl;
            log_file << "\tInitial training loss: " << fixed << setprecision(4) << attempt_result->initial_train_loss << endl;
            log_file << "\tBest training loss: " << fixed << setprecision(4) << attempt_result->best_train_loss << endl;
            log_file << "\tEpochs: " << attempt_result->epochs << endl;
            log_file << "\tTime: " << attempt_result->training_time_ms << endl;
            log_file << "\tLearning rate adjustment factor: " << learningRateAdjustmentFactor << endl;
            log_file << "\tComplexity modifier: " << complexity_modifier << endl;
            log_file << "Parameters: " << neuralNetwork->get_total_parameters() << endl;
            log_file << "Batch size: " << max_batch_size << endl;
            log_file << "Learning rate: " << attempt_result->learning_rate << endl;
            log_file << "Trained: " << (attempt_result->trained ? "true" : "false") << endl;
            log_file << "Generalized: " << (attempt_result->generalized ? "true" : "false") << endl;
            log_file << "Accuracy: " << fixed << setprecision(2) << (attempt_result->accuracy * 100) << "%" << endl;
        } catch (const exception &e) {
            attempt_result = make_shared<TrainingResult>();
        }
        return attempt_result;
    }

    bool create_label_task(const string &task_name, const string &goal, const string &dataset_name,
                           const string &dataset_file_path, const string &task_folder_path,
                           const string &test_dataset_file_path) {
        // TODO: this entire process could be more robust. Right now, it takes a very naive approach
        //  by handling images, labels, and numbers with the same simple architecture.
        //  I think it should look at the given inputs and outputs, estimate a model complexity,
        //  pick an architecture, and then shape the network to match the inputs and outputs.
        //  This initial code can make models that are considerably larger than needed, which might
        //  perform really poorly in some situations. It could also make models that are too small
        //  to be useful in other situations.
        try {
            string log_file_path = task_folder_path + "/../logs/log.txt";
            filesystem::create_directories(task_folder_path + "/../logs");
            ofstream log_file(log_file_path, ios_base::app);

            string task_full_path = task_folder_path + task_name;
            if (filesystem::exists(task_full_path)) {
                string config = task_full_path + "/model.config";
                if (filesystem::exists(config)) {
                    cout << "Task " << task_name << " already exists. Skipping." << endl;
                    return true;
                }
                cout << "Task " << task_name << " already exists, but is incomplete. Removing." << endl;
                filesystem::remove_all(task_full_path);
            }
            cout << "Creating label task " << task_name << " with goal " << goal << " using dataset " << dataset_name << endl;
            log_file << "Creating label task " << task_name << " with goal " << goal << " using dataset " << dataset_name << endl;

            string dataset_full_file_path = dataset_file_path + "/dataset.bin";
            shared_ptr<BinaryDataSet> dataSource;
            shared_ptr<BinaryDataSet> testDataSource;

            if (!test_dataset_file_path.empty()) {
                dataSource = make_shared<BinaryDataSet>(dataset_full_file_path);
                string test_dataset_full_file_path = test_dataset_file_path + "/dataset.bin";
                testDataSource = make_shared<BinaryDataSet>(test_dataset_full_file_path,
                                                            dataSource->getGivenMetadata(),
                                                            dataSource->getExpectedMetadata());
            } else {
                dataSource = make_shared<BinaryDataSet>(dataset_full_file_path, 0.9);
                testDataSource = make_shared<BinaryDataSet>(dataset_full_file_path, -0.1);
            }

            BinaryDatasetReader reader(dataset_full_file_path);

            int max_batch_size = std::min(128, (int) dataSource->recordCount());

            OptimizerType optimizerType;
            if (goal == "memory") {
                optimizerType = OptimizerType::sgd;
            } else {
                optimizerType = OptimizerType::adam;
            }
            ActivationType activationType = ActivationType::leaky;
            bool multiple_outputs = dataSource->getExpectedShapes().size() > 1;
            LossType lossType;
            if (!multiple_outputs && goal != "speed") {
                if (dataSource->getExpectedShape()[0] == 1 &&
                    dataSource->getExpectedShape()[1] == 1 &&
                    dataSource->getExpectedShape()[2] == 1) {
                    lossType = LossType::binaryCrossEntropy;
                } else {
                    lossType = LossType::categoricalCrossEntropy;;
                }
            } else {
                lossType = LossType::mse;
            }

            bool found = false;
            float learningRateSearchRate = ("speed" == goal) ? 2.0f : 1.0f;
            size_t patience;
            size_t min_epochs;
            float improvement_tolerance;
            if (dataSource->recordCount() > 999) {
                patience = ("speed" == goal) ? 1 : 2;
                min_epochs = 2;
                improvement_tolerance = 1e-5;
            } else if (dataSource->recordCount() < 100) {
                patience = ("speed" == goal) ? 10 : 20;
                min_epochs = patience * 2;
                improvement_tolerance = 1e-7;
            } else {
                // 100 - 999
                patience = ("speed" == goal) ? 5 : 10;
                min_epochs = patience * 5;
                improvement_tolerance = 1e-7;
            }
            std::random_device rd;
            std::mt19937 gen(rd());
//            std::uniform_int_distribution<> dis(1, 100);

            vector<shared_ptr<RawDecoder >> expected_decoders = build_expected_decoders(false, reader);
            float target_accuracy = 0.90;
            shared_ptr<NeuralNetworkForTraining> neuralNetwork;
            ElapsedTimer timer;

            // I suspect there are multiple viable combinations, but that they are likely grouped together near each other.
            // in order to make it more likely to find them sooner, I will shuffle the possibilities with the hope of making
            // one of those viable options closer to the beginning.
            vector<pair<double, float>> complexity_lr_pairs;
            {
                float max_learning_rate_adjustment_factor = "speed" == goal ? 8 : 15.0;
                float max_complexity_modifier = "speed" == goal ? 8.0 : 12.0;
                double complexity_modifier = 1.0;
                while (complexity_modifier <= max_complexity_modifier) {
                    float learningRateAdjustmentFactor = "speed" == goal ? 4.0 : 0.0f;
                    while (learningRateAdjustmentFactor <= max_learning_rate_adjustment_factor) {
                        complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor);
                        learningRateAdjustmentFactor += learningRateSearchRate;
                    }
                    complexity_modifier *= 1.25;
                }
                std::shuffle(complexity_lr_pairs.begin(), complexity_lr_pairs.end(), gen);
            }
            cout << endl << "Searching for training parameters that work for " << task_name << "." << endl;
            cout << "Using optimizer " << optimizerTypeToString(optimizerType) << endl;
            cout << "Using loss function " << lossTypeToString(lossType) << endl;
            cout << "Using activation function " << activationTypeToString(activationType) << endl;
            cout << "Using batch size " << max_batch_size << endl;
            int attempt = 0;
            float next_search_size = 640.0f;

            while (!found) {
                int retries = 0;
                float best_accuracy = 0.0f;
                string best_attempt_key;
                shared_ptr<BinaryDataSet> searchDataSource;
                shared_ptr<BinaryDataSet> testSearchDataSource;
                bool using_source = (next_search_size + 1.0f) >= (float) dataSource->recordCount();
                if (!using_source) {
                    float search_ratio = next_search_size / (float) dataSource->recordCount();
                    searchDataSource = make_shared<BinaryDataSet>(dataset_full_file_path, search_ratio);
                    if (!test_dataset_file_path.empty()) {
                        string test_dataset_full_file_path = test_dataset_file_path + "/dataset.bin";
                        testSearchDataSource = make_shared<BinaryDataSet>(test_dataset_full_file_path, dataSource->getGivenMetadata(),
                                                                          dataSource->getExpectedMetadata(), search_ratio);
                    } else {
                        testSearchDataSource = make_shared<BinaryDataSet>(dataset_full_file_path, -search_ratio);
                    }
                } else {
                    searchDataSource = dataSource;
                    testSearchDataSource = testDataSource;
                }
                vector<shared_ptr<TrainingResult>> next_results;
                cout << "Checking " << complexity_lr_pairs.size() << " combinations." << endl;
                log_file << "Checking " << complexity_lr_pairs.size() << " combinations." << endl;
                int experiment_count = 0;
                for (auto complexity_lr_pair: complexity_lr_pairs) {
                    double complexity_modifier = complexity_lr_pair.first;
                    float learningRateAdjustmentFactor = complexity_lr_pair.second;
                    attempt++;
                    experiment_count++;
                    cout << "__________________Experiment: (" << attempt << ") " << experiment_count << " of " << complexity_lr_pairs.size() << "__________________" << endl;
                    log_file << "__________________Experiment: (" << attempt << ") " << experiment_count << " of " << complexity_lr_pairs.size() << "__________________" << endl;
                    string attempt_key = getAttemptKey(max_batch_size, learningRateAdjustmentFactor, complexity_modifier);
                    auto exit_strategy = make_shared<DefaultExitStrategy>((using_source ? patience * 2 : patience),
                                                                          NINETY_DAYS_MS,
                                                                          1000000,
                                                                          1e-3,
                                                                          (using_source ? improvement_tolerance / 10.0f : improvement_tolerance),
                                                                          (using_source ? min_epochs * 2 : min_epochs),
                                                                          (using_source ? 0.25f : 0.05f));
                    shared_ptr<TrainingResult> attempt_result = attempt_training_with_logging(task_name, goal, task_folder_path, testSearchDataSource,
                                                                                              max_batch_size, optimizerType, activationType,
                                                                                              lossType, learningRateSearchRate, patience,
                                                                                              min_epochs, improvement_tolerance, exit_strategy,
                                                                                              expected_decoders, target_accuracy, attempt,
                                                                                              complexity_modifier, learningRateAdjustmentFactor, log_file,
                                                                                              searchDataSource, reader, neuralNetwork);
                    next_results.push_back(attempt_result);
                    if (attempt_result->accuracy > best_accuracy) {
                        best_accuracy = attempt_result->accuracy;
                        best_attempt_key = attempt_key;
                    }
                    if (using_source && attempt_result->accuracy >= target_accuracy) {
                        found = true;
                        cout << "Model is accurate." << endl;
                        log_file << "Model is accurate." << endl;
                        break;
                    }
                }
                if (!found) {
                    cout << "Best accuracy: " << (best_accuracy * 100) << "%" << endl;
                    log_file << "Best accuracy: " << (best_accuracy * 100) << "%" << endl;
                    // sort on accuracy with the lowest test loss tie-breaking
                    std::sort(next_results.begin(), next_results.end(), [](shared_ptr<TrainingResult> a, shared_ptr<TrainingResult> b) {
                        return a->accuracy > b->accuracy || (a->accuracy == b->accuracy && a->final_test_loss < b->final_test_loss);
                    });
                    cout << "______________________LEADER BOARD______________________" << endl;
                    for (const auto &result: next_results) {
                        cout << "accuracy: " << (result->accuracy * 100) << "%, complexity: " << result->complexity_modifier << ", learning rate: " << result->learningRateAdjustmentFactor << endl;
                        log_file << "accuracy: " << (result->accuracy * 100) << "%, complexity: " << result->complexity_modifier << ", learning rate: " << result->learningRateAdjustmentFactor << endl;
                    }
                    cout << endl;

                    if (next_results.size() > 1) {
                        int top_results;
                        if (using_source) {
                            top_results = 1;
                        } else {
                            top_results = std::max((int) next_results.size() / 4, 2);
                        }
                        cout << "Keeping: " << top_results << " of " << next_results.size() << " complexity/learning rate pairs." << endl;
                        log_file << "Keeping: " << top_results << " of " << next_results.size() << " complexity/learning rate pairs." << endl;
                        next_results.erase(next_results.begin() + top_results, next_results.end());
                        cout << "Leader board kept: " << next_results.size() << endl;
                        vector<pair<double, float>> next_complexity_lr_pairs;
                        for (auto complexity_lr_pair: complexity_lr_pairs) {
                            double complexity_modifier = complexity_lr_pair.first;
                            float learningRateAdjustmentFactor = complexity_lr_pair.second;
                            for (const auto &result: next_results) {
                                if (result->complexity_modifier == complexity_modifier && result->learningRateAdjustmentFactor == learningRateAdjustmentFactor) {
                                    next_complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor);
                                    break;
                                }
                            }
                        }
                        complexity_lr_pairs = next_complexity_lr_pairs;
                        cout << "Complexity/learning rate pairs left: " << complexity_lr_pairs.size() << endl;
                        log_file << "Complexity/learning rate pairs left: " << complexity_lr_pairs.size() << endl;
                        for (auto next_pair: complexity_lr_pairs) {
                            cout << "Complexity: " << next_pair.first << ", learning rate: " << next_pair.second << endl;
                            log_file << "Complexity: " << next_pair.first << ", learning rate: " << next_pair.second << endl;
                        }
                    } else if (using_source) {
                        if (retries < 3) {
                            retries++;
                            vector<pair<double, float>> next_complexity_lr_pairs;
                            double complexity_modifier = complexity_lr_pairs.front().first;
                            float learningRateAdjustmentFactor = complexity_lr_pairs.front().second;
                            next_complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier / 1.1, learningRateAdjustmentFactor);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier * 2.0, learningRateAdjustmentFactor);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier / 2.0, learningRateAdjustmentFactor);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier / 1.1, learningRateAdjustmentFactor - 0.5f);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor - 0.5f);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor - 2.5f);
                            next_complexity_lr_pairs.emplace_back(complexity_modifier, learningRateAdjustmentFactor / 2.0f);
                            complexity_lr_pairs = next_complexity_lr_pairs;
                            cout << "Didn't generalize, trying with a lower learning rate and lower complexity." << endl;
                        } else {
                            cout << "Only one complexity/learning rate pair left.  Using it." << endl;
                            log_file << "Only one complexity/learning rate pair left.  Using it." << endl;
                            found = true;
                        }
                    }
                }
                if (!found) {
                    if (complexity_lr_pairs.size() > 2 && next_search_size < (float) dataSource->recordCount()) {
                        next_search_size *= 2.0f;
                    } else {
                        next_search_size = (float) dataSource->recordCount();
                    }
                }
            }

            neuralNetwork->saveWithOverwrite();

            string task_dataset_metadata_path = task_full_path + "/dataset.bin";
            BinaryDatasetWriter writer(task_dataset_metadata_path, reader.get_given_metadata(), reader.get_expected_metadata());
            writer.close();
            reader.close();
            log_file.close();
        } catch (exception &e) {
            cout << "Error: " << e.what() << endl;
            return false;
        }

        return true;
    }

    bool create_happyml_task(const string &task_type, const string &task_name, const string &goal, const string &dataset_name,
                             const string &dataset_file_path, const string &task_folder_path, const string &test_dataset_file_path) {
        // TODO: we need to cache the task in the context so we don't have to reload it.
        if (task_type == "label") {
            return create_label_task(task_name, goal, dataset_name, dataset_file_path, task_folder_path, test_dataset_file_path);
        }
        cout << "Unknown task type " << task_type << endl;
        return false;
    }

}
#endif //HAPPYML_TASK_UTILS_HPP
