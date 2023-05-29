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

    size_t estimate_layer_output_size(const vector<vector<size_t>> given_shapes,
                                      const vector<vector<size_t>> &expected_shapes,
                                      const string &goal) {
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
        size_t max_val = total_output_values * 8;
        total_values = std::min(total_values, max_val);
        if (goal == "speed") {
            total_values = total_values / 4;
        }
        return total_values;
    }

    size_t estimate_layer_output_size(vector<vector<size_t>> expected_shapes, const string &goal) {
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
        if (goal != "speed") {
            total_values = total_values * 2;
        }
        return total_values;

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

        //vector<string> merged_headers = pretty_print_merge_headers(expected_column_names, given_column_names);
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
        //auto merged_values = pretty_print_merge_records(expected_decoders, predictions, given_decoders, given_values);
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
                                                                        int attempt,
                                                                        OptimizerType optimizerType,
                                                                        LossType lossType,
                                                                        ActivationType activationType) {

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
        if (lossType == LossType::categoricalCrossEntropy) {
            // categorical cross entropy is very sensitive to learning rate
            learningRate *= 0.1f;
            biasLearningRate *= 0.1f;
        }
        const float decay_rate = 0.95f;
        learningRate *= std::powf(decay_rate, (float) attempt);
        biasLearningRate *= std::powf(decay_rate, (float) attempt);
        cout << std::fixed << std::setprecision(6) << "Using learning rate " << learningRate << endl;
        cout << "Using bias learning rate " << biasLearningRate << endl;

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
                                                                       dataSource->getExpectedShapes(), goal);
        if (dataSource->getGivenShapes().size() > 1) {
            shared_ptr<HappymlDSL::NNVertex> input_layer = initial_layers->add_concatenated_input_layer(dataSource->getGivenShapes());
            layer1 = input_layer
                    ->addLayer(output_size_given_expected,
                               LayerType::full, activationType);
            cout << "Using concatenated input layer (" << dataSource->getGivenShapes().size() << ")" << endl;
        } else {
            is_convolutional = reader.get_given_metadata(0)->purpose == 'I';
            if (is_convolutional) {
                layer1 = initial_layers->addInputLayer(dataSource->getGivenShape(), 1, 3, LayerType::convolution2dValid,
                                                       activationType);
                cout << "Using convolutional input layer (1 filter and kernel size 3)" << endl;
            } else {
                layer1 = initial_layers->addInputLayer(dataSource->getGivenShape(),
                                                       output_size_given_expected,
                                                       LayerType::full, activationType);
                cout << "Using full input layer: " << output_size_given_expected << endl;
            }
        }
        layer1->setUseL2Regularization(true);
        layer1->setUseNormClipping(true);
        if (lossType == LossType::categoricalCrossEntropy) {
            layer1->setUseNormalization(true);
            cout << "Using normalization layer" << endl;
        }
        if (goal == "memory") {
            layer1 = layer1->setBits(8)->setMaterialized(false);
        }
        shared_ptr<HappymlDSL::NNVertex> layer2;
        if (is_convolutional) {
            layer2 = layer1
                    ->addLayer(output_size_given_expected,
                               LayerType::full, activationType);
            layer2->setUseL2Regularization(true);
            layer2->setUseNormClipping(true);
            cout << "Using full second layer: " << output_size_given_expected << endl;
        } else {
            size_t output_size_expected = estimate_layer_output_size(dataSource->getExpectedShapes(), goal);
            layer2 = layer1
                    ->addLayer(output_size_expected,
                               LayerType::full, activationType);
            layer2->setUseL2Regularization(true);
            layer2->setUseNormClipping(true);
            cout << "Using full second layer: " << output_size_expected << endl;
        }
        if (lossType == LossType::categoricalCrossEntropy) {
            layer2->setUseNormalization(true);
            cout << "Using normalization layer" << endl;
        }

        if (goal == "memory") {
            layer2 = layer2->setBits(8)->setMaterialized(false);
        }

        ActivationType last_activation;
        if (lossType == LossType::categoricalCrossEntropy) {
            last_activation = ActivationType::softmax;
        } else {
            last_activation = ActivationType::sigmoid;
        }
        for (int i = 0; i < dataSource->getExpectedShapes().size(); i++) {
            auto next_expected_shape = dataSource->getExpectedShapes()[i];
            auto output_layer = layer2
                    ->addOutputLayer(next_expected_shape,
                                     last_activation)
                    ->setUseBias(true);
            cout << "Using output layer: " << dataSource->getExpectedShape()[0] << ", " << dataSource->getExpectedShape()[1] << ", " << dataSource->getExpectedShape()[2] << endl;
        }
        auto neuralNetwork = neural_network_builder->build();
        return neuralNetwork;
    }

    shared_ptr<TrainingResult> training_test(const string &task_name, const string &goal, const string &task_folder_path, const shared_ptr<BinaryDataSet> &search_data_source, int batch_size, int attempt, OptimizerType &optimizerType,
                                             LossType &lossType, shared_ptr<BinaryDataSet> &dataSource, BinaryDatasetReader &reader, ActivationType &activationType) {
        cout << endl << "Searching for training parameters that works for " << task_name << "." << endl;
        cout << "Using batch size " << batch_size << endl;
        cout << "Using optimizer " << optimizerTypeToString(optimizerType) << endl;
        cout << "Using loss function " << lossTypeToString(lossType) << endl;
        cout << "Using activation function " << activationTypeToString(activationType) << endl;
        shared_ptr<NeuralNetworkForTraining> neuralNetwork = build_neural_network_for_label(task_name, task_folder_path, goal, dataSource, reader, attempt, optimizerType, lossType, activationType);
        neuralNetwork->useTestPrecisionExitStrategy();
        try {
            return neuralNetwork->train(search_data_source, batch_size);
        } catch (const std::exception &e) {
            shared_ptr<TrainingResult> result = make_shared<TrainingResult>();
            return result;
        }
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

            string dataset_full_file_path = dataset_file_path + "/dataset.bin";
            auto dataSource = make_shared<BinaryDataSet>(dataset_full_file_path);
            shared_ptr<BinaryDataSet> testDataSource = nullptr;
            shared_ptr<BinaryDataSet> search_data_source = dataSource;
            if (!test_dataset_file_path.empty()) {
                string test_dataset_full_file_path = test_dataset_file_path + "/dataset.bin";
                testDataSource = make_shared<BinaryDataSet>(test_dataset_full_file_path,
                                                            dataSource->getGivenMetadata(),
                                                            dataSource->getExpectedMetadata());
                search_data_source = testDataSource;
            }
            BinaryDatasetReader reader(dataset_full_file_path);

            int batch_size;
            if (goal != "speed") {
                batch_size = 32;
            } else {
                batch_size = 64;
            }
            if (dataSource->recordCount() < batch_size) {
                batch_size = 1;
            }
            OptimizerType optimizerType;
            if (goal == "memory") {
                optimizerType = OptimizerType::sgd;
            } else {
                optimizerType = OptimizerType::adam;
            }
            LossType lossType = LossType::categoricalCrossEntropy;
            ActivationType activationType = ActivationType::relu;
            bool multiple_outputs = dataSource->getExpectedShapes().size() > 1;
            bool found = false;
            int attempt;
            float loss_epsilon = 0.01f;
            if (!multiple_outputs && goal != "speed") {
                // categorical is so much slower than mse, so we'll skip categorical when training speed is needed.
                // cross entropy with multiple outputs may not work, so we'll only try it with a single output
                if (dataSource->getExpectedShape()[0] == 1 &&
                    dataSource->getExpectedShape()[1] == 1 &&
                    dataSource->getExpectedShape()[2] == 1) {
                    lossType = LossType::binaryCrossEntropy;
                }
                shared_ptr<TrainingResult> attempt_result;
                for (attempt = 0; attempt < 10; attempt++) {
                    activationType = ActivationType::leaky;
                    attempt_result = training_test(task_name, goal, task_folder_path, search_data_source, batch_size, attempt, optimizerType, lossType, dataSource, reader, activationType);
                    if (attempt_result->final_loss + loss_epsilon < attempt_result->initial_loss) {
                        found = true;
                        break;
                    }
                    activationType = ActivationType::relu;
                    attempt_result = training_test(task_name, goal, task_folder_path, search_data_source, batch_size, attempt, optimizerType, lossType, dataSource, reader, activationType);
                    if (attempt_result->final_loss + loss_epsilon < attempt_result->initial_loss) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                lossType = LossType::mse;
                shared_ptr<TrainingResult> attempt_result;
                for (attempt = 0; attempt < 10; attempt++) {
                    if (goal != "speed") {
                        // if we need to find any configuration that works well enough, we'll skip leaky relu
                        activationType = ActivationType::leaky;
                        attempt_result = training_test(task_name, goal, task_folder_path, search_data_source, batch_size, attempt, optimizerType, lossType, dataSource, reader, activationType);
                        if (attempt_result->final_loss + loss_epsilon < attempt_result->initial_loss) {
                            found = true;
                            break;
                        }
                    }
                    activationType = ActivationType::relu;
                    attempt_result = training_test(task_name, goal, task_folder_path, search_data_source, batch_size, attempt, optimizerType, lossType, dataSource, reader, activationType);
                    if (attempt_result->final_loss + loss_epsilon < attempt_result->initial_loss) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                cout << "Unable to create a model that trains. Giving up." << endl;
                return false;
            }

            cout << "Found training parameters that works for " << task_name << "." << endl;
            cout << "Using batch size " << batch_size << endl;
            cout << "Using optimizer " << optimizerTypeToString(optimizerType) << endl;
            cout << "Using loss function " << lossTypeToString(lossType) << endl;
            cout << "Using activation function " << activationTypeToString(activationType) << endl;
            auto neuralNetwork = build_neural_network_for_label(task_name, task_folder_path, goal, dataSource, reader, attempt, optimizerType, lossType, activationType);

            if (goal != "speed") {
                neuralNetwork->useHighPrecisionExitStrategy();
            }
            float loss;
            if (testDataSource == nullptr) {
                loss = neuralNetwork->train(dataSource, batch_size)->final_loss;
            } else {
                loss = neuralNetwork->train(dataSource, testDataSource, batch_size)->final_loss;
            }

            cout << fixed << setprecision(4) << "Loss: " << loss << endl;
            neuralNetwork->saveWithOverwrite();

            dataSource->restart();
            vector<shared_ptr<RawDecoder >> expected_decoders = build_expected_decoders(false, reader);
            float accuracy = neuralNetwork->compute_categorical_accuracy(dataSource, expected_decoders);
            cout << "Accuracy: " << fixed << setprecision(4) << accuracy << endl;
            string task_dataset_metadata_path = task_full_path + "/dataset.bin";
            BinaryDatasetWriter writer(task_dataset_metadata_path, reader.get_given_metadata(), reader.get_expected_metadata());
            writer.close();
            reader.close();
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
