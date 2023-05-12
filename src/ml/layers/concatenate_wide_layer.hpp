//
// Created by Erik Hyrkas on 5/8/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CONCATINATE_WIDE_LAYER_HPP
#define HAPPYML_CONCATINATE_WIDE_LAYER_HPP

#include <utility>

#include "../neural_network_layer_function.hpp"
#include "../../types/tensor_views/tensor_concat_wide_view.h"
#include "../../types/tensor_views/tensor_window_view.h"

namespace happyml {
    class ConcatenateWideLayer : public happyml::NeuralNetworkLayerFunction {
    public:
        explicit ConcatenateWideLayer(string label, vector<vector<size_t>> input_shapes) : label_(std::move(label)), input_shapes_(input_shapes) {
            // vector of shapes (rows, columns, channels)
            // all shapes must have the same number of rows and channels
            size_t rows = input_shapes[0][0];
            size_t combined_columns = 0;
            size_t channels = input_shapes[0][2];
            if (input_shapes.size() < 2) {
                throw runtime_error("ConcatenateWideLayer: input must have at least 2 tensors.");
            }

            for (auto next: input_shapes) {
                if (next.size() != 3) {
                    throw runtime_error("ConcatenateWideLayer: input shape must have 3 dimensions.");
                }
                if (next[0] != rows) {
                    throw runtime_error("ConcatenateWideLayer: all input shapes must have the same number of rows.");
                }
                if (next[2] != channels) {
                    throw runtime_error("ConcatenateWideLayer: all input shapes must have the same number of channels.");
                }
                combined_columns += next[1];
            }
            output_shape_ = {rows, combined_columns, channels};
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            if (input.size() < 2) {
                throw runtime_error("You need at least two tensors to concatenate.");
            }

            shared_ptr<BaseTensor> result = make_shared<TensorConcatWideView>(input[0], input[1]);

            for (size_t i = 2; i < input.size(); ++i) {
                result = make_shared<TensorConcatWideView>(result, input[i]);
            }

            return result;
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) override {
            vector<shared_ptr<BaseTensor>> errors;
            size_t start_column = 0;

            for (const auto& shape : input_shapes_) {
                size_t input_column_count = shape[1];
                auto error_view = make_shared<TensorWindowView>(output_error, start_column, start_column + input_column_count);
                errors.push_back(error_view);
                start_column += input_column_count;
            }

            return errors;
        }

        vector<size_t> get_output_shape() {
            return output_shape_;
        }

    private:
        vector<vector<size_t>> input_shapes_;
        vector<size_t> output_shape_;
        string label_;
    };
}
#endif //HAPPYML_CONCATINATE_WIDE_LAYER_HPP
