//
// Created by Erik Hyrkas on 6/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HYPERBAND_SPACE_HPP
#define HAPPYML_HYPERBAND_SPACE_HPP

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace happyml {
    struct Hyperparameters {
        float learning_rate;
        float bias_learning_rate;
        int complexity_depth;
        float complexity_width;
        float dropout_rate;
        float l2_regularization_strength;
        int batch_size;
        bool use_normalization_layers;
        bool use_bias;
        bool use_hidden_bias;
        bool use_normal_clipping;

        int bits; // how many bits are used to represent each parameter?
        float evaluation_metric; // how did these parameters perform?
        bool minimize_metric;   // do we want a small or big evaluationMetric?

        [[nodiscard]] std::string as_string() const {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(8);
            ss << "bits: " << bits << ", ";
            ss << "learning_rate: " << learning_rate << ", ";
            ss << "bias_learning_rate: " << bias_learning_rate << ", ";
            ss << "complexity_depth: " << complexity_depth << ", ";
            ss << "complexity_width: " << complexity_width << ", ";
            ss << "dropout_rate: " << dropout_rate << ", ";
            ss << "l2_regularization_strength: " << l2_regularization_strength << ", ";
            ss << "batch_size: " << batch_size << ", ";
            ss << "use_normalization_layers: " << use_normalization_layers << ", ";
            ss << "use_bias: " << use_bias << ", ";
            ss << "use_hidden_bias: " << use_hidden_bias << ", ";
            ss << "use_normal_clipping: " << use_normal_clipping;
            return ss.str();
        }

        [[nodiscard]] std::string temp_folder_name() const {
            std::hash<std::string> hasher;
            std::size_t hash_value = hasher(as_string());

            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

            std::stringstream ss;
            ss << std::hex << hash_value << "_" << timestamp;
            return ss.str();
        }
    };

    class HyperparameterSpace {
    public:
        explicit HyperparameterSpace(int max_batch_sizes) {
            learning_rate_space = {0.001, 0.0007, 0.0005, 0.0003, 0.0001, 0.00007, 0.00005, 0.00003, 0.005, 0.01f, 0.003};
            bias_learning_rate_space = {0.0001, 0.00007, 0.00005, 0.00003, 0.001, 0.0007, 0.0005, 0.0003, 0.01f, 0.005, 0.003};
            complexity_depth_space = {3, 2, 1, 5, 7, 10, 20, 30};
            complexity_width_space = {1.0f, 2.0f, 5.0f, 10.0f, 7.0f, 3.0f, 1.7f, 1.5f, 1.4f, 1.3f, 1.2f, 1.1f, 20.0f, 30.0f};
            dropout_rate_space = {0.8f, 0.5f, 0.0f, 0.7f, 0.6f, 0.4f, 0.3f};
            l2_regularization_strength_space = {0.02f, 0.015f, 0.01f, 0.0f, 0.005, 0.001, 0.0001, 0.05f, 0.04f, 0.035f, 0.03f, 0.025f, 0.1f, 0.2f};
            batch_size_space = {32, 64, 128, 1, 256, 512, 1024};
            // remove batch sizes that are too big for the dataset
            for (int i = 0; i < batch_size_space.size(); i++) {
                if (batch_size_space[i] > max_batch_sizes && batch_size_space[i] > 1) {
                    batch_size_space.erase(batch_size_space.begin() + i);
                    i--;
                }
            }

            use_normalization_layers_space = {false, true};
            use_hidden_bias_space = {false, true};
            use_bias_space = {true, false};
            use_normal_clipping_space = {false, true};
        }

        std::vector<float> learning_rate_space;
        std::vector<float> bias_learning_rate_space;
        std::vector<int> complexity_depth_space; // how many layers or components are there
        std::vector<float> complexity_width_space; // how wide are the layers with regard to the inputs
        std::vector<float> dropout_rate_space;
        std::vector<float> l2_regularization_strength_space;
        std::vector<int> batch_size_space;
        std::vector<bool> use_normalization_layers_space;
        std::vector<bool> use_hidden_bias_space;
        std::vector<bool> use_bias_space;
        std::vector<bool> use_normal_clipping_space;

        [[nodiscard]] size_t getNumConfigurations() const {
            return learning_rate_space.size() * bias_learning_rate_space.size() * complexity_depth_space.size() *
                   complexity_width_space.size() * dropout_rate_space.size() * l2_regularization_strength_space.size() *
                   batch_size_space.size() * use_normalization_layers_space.size() * use_hidden_bias_space.size() *
                   use_bias_space.size() * use_normal_clipping_space.size();
        }
    };

}
#endif //HAPPYML_HYPERBAND_SPACE_HPP
