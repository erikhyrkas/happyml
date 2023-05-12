//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_NEURAL_NETWORK_LAYER_FUNCTION_HPP
#define HAPPYML_NEURAL_NETWORK_LAYER_FUNCTION_HPP

#include "activation.hpp"
#include "optimizer.hpp"
#include "../types/tensor_views/tensor_flatten_to_row_view.hpp"
#include "../types/tensor_views/tensor_transpose_view.hpp"
#include "../types/tensor_views/tensor_sum_to_channel_view.hpp"
#include "../types/tensor_views/tensor_full_convolve_2d_view.hpp"
#include "../types/tensor_views/tensor_sum_channels_view.hpp"
#include "../types/tensor_views/tensor_channel_to_tensor_view.hpp"
#include "../types/tensor_views/tensor_reshape_view.hpp"
#include "../types/tensor_views/tensor_matrix_multiply_tensor_view.hpp"

namespace happyml {
    // side note: I read an article on back propagation I thought was interesting:
    // https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d

    class NeuralNetworkLayerFunction {
    public:
        virtual shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) = 0;

        virtual vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) = 0;

        virtual void saveKnowledge(const string &fullKnowledgePath) {
        }

        virtual void loadKnowledge(const string &fullKnowledgePath) {
        }
    };


}
#endif //HAPPYML_NEURAL_NETWORK_LAYER_FUNCTION_HPP
