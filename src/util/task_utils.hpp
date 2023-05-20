//
// Created by Erik Hyrkas on 5/14/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TASK_UTILS_HPP
#define HAPPYML_TASK_UTILS_HPP

#include "dataset_utils.hpp"
#include "../ml/happyml_dsl.hpp"

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
        } catch (exception &e) {
            cout << "Error: " << e.what() << endl;
            return false;
        }
        return true;
    }

    bool create_happyml_task(const string &task_type, const string &task_name, const string &goal, const string &dataset_name,
                             const string &dataset_file_path, const string &task_folder_path, const string &test_dataset_file_path) {
        if (task_type == "label") {
            return create_label_task(task_name, goal, dataset_name, dataset_file_path, task_folder_path, test_dataset_file_path);
        }
        cout << "Unknown task type " << task_type << endl;
        return false;
    }

}
#endif //HAPPYML_TASK_UTILS_HPP
