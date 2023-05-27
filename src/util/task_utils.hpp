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

    size_t estimate_first_layer_label_task_size(vector<vector<size_t>> given_shapes, const string &goal) {
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
        if (goal == "speed") {
            total_values = total_values / 2;
        }
        return total_values;
    }

    size_t estimate_second_layer_label_task_size(vector<vector<size_t>> expected_shapes, const string &goal) {
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

        vector<string> merged_headers = pretty_print_merge_headers(expected_column_names, given_column_names);
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

        auto merged_values = pretty_print_merge_records(expected_decoders, predictions, given_decoders, given_values);
        if (widths.empty()) {
            // using width of first result is suboptimal, but it's good enough for now.
            widths = calculate_pretty_print_column_widths(merged_headers, merged_values);
            pretty_print_header(cout, merged_headers, widths);
        }
        pretty_print_row(cout, merged_values, widths);


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
        reader.close();

        vector<string> merged_headers = pretty_print_merge_headers(expected_column_names, given_column_names);
        vector<size_t> widths;
        cout << "Results: " << endl;
        auto dataset = make_shared<BinaryDataSet>(dataset_full_file_path);
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

    bool create_label_task(const string &task_name, const string &goal, const string &dataset_name,
                           const string &dataset_file_path, const string &task_folder_path,
                           const string &test_dataset_file_path) {
        try {
            string task_full_path = task_folder_path + task_name;
            if (filesystem::exists(task_full_path)) {
                cout << "Task " << task_name << " already exists. Skipping." << endl;
                return true;
            }
            cout << "Creating label task " << task_name << " with goal " << goal << " using dataset " << dataset_name << endl;
//            cout << "Dataset file path: " << dataset_file_path << endl;
//            if (!test_dataset_file_path.empty()) {
//                cout << "Test dataset file path: " << test_dataset_file_path << endl;
//            }
//            cout << "Task folder path: " << task_folder_path << endl;


            string dataset_full_file_path = dataset_file_path + "/dataset.bin";
            auto dataSource = make_shared<BinaryDataSet>(dataset_full_file_path);

            auto nnbuilder = neuralNetworkBuilder();
            auto initial_layers = nnbuilder
                    ->setModelName(task_name)
                    ->setModelRepo(task_folder_path)
                    ->setLossFunction(LossType::categoricalCrossEntropy)
                    ->add_concatenated_input_layer(dataSource->getGivenShapes());

            auto dense_layer1 = initial_layers
                    ->addLayer(estimate_first_layer_label_task_size(dataSource->getGivenShapes(), goal),
                               LayerType::full, ActivationType::leaky);
            if (goal == "memory") {
                dense_layer1 = dense_layer1->setBits(8)->setMaterialized(false);
            }
            auto dense_layer2 = dense_layer1
                    ->addLayer(estimate_second_layer_label_task_size(dataSource->getExpectedShapes(), goal),
                               LayerType::full, ActivationType::leaky);

            if (goal == "memory") {
                dense_layer2 = dense_layer2->setBits(8)->setMaterialized(false);
            }

            auto neuralNetwork = dense_layer2
                    ->addOutputLayer(dataSource->getExpectedShape(),
                                     ActivationType::softmax)
                    ->setUseBias(true)
                    ->build();

            neuralNetwork->useHighPrecisionExitStrategy();
            int batch_size = 32;
            if (dataSource->recordCount() < batch_size) {
                batch_size = 1;
            }
            float loss;
            if (test_dataset_file_path.empty()) {
                loss = neuralNetwork->train(dataSource, batch_size);
            } else {
                string test_dataset_full_file_path = test_dataset_file_path + "/dataset.bin";
                auto testDataSource = make_shared<BinaryDataSet>(test_dataset_full_file_path);
                loss = neuralNetwork->train(dataSource, testDataSource, batch_size);
            }

            cout << fixed << setprecision(4) << "Loss: " << loss << endl;
            neuralNetwork->saveWithOverwrite();

            dataSource->restart();
            BinaryDatasetReader reader(dataset_full_file_path);
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
