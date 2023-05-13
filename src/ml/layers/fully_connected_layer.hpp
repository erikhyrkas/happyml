//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FULLY_CONNECTED_LAYER_HPP
#define HAPPYML_FULLY_CONNECTED_LAYER_HPP

#include "../neural_network_layer_function.hpp"
#include "../../types/tensor_impls/tensor_from_xavier.hpp"

namespace happyml {
    class FullyConnectedLayer : public NeuralNetworkLayerFunction {
    public:
        FullyConnectedLayer(const string &label, size_t inputSize, size_t outputSize, uint8_t bits,
                            const shared_ptr<BaseOptimizer> &optimizer) {
            this->label = label;
            this->registration_id = optimizer->registerForWeightChanges();
            this->inputShapes = vector<vector<size_t >>{{1, inputSize, 1}};
            this->outputShape = vector<size_t>{1, outputSize, 1};
            this->weights = make_shared<TensorFromXavier>(inputSize, outputSize, 1, 42);
            this->bits = bits;
            this->optimizer = optimizer;
            if (bits == 32) {
                mixedPrecisionScale = 0.5f;
            } else if (bits == 16) {
                mixedPrecisionScale = 2.f;
            } else {
                mixedPrecisionScale = 3.f;
            }
        }

        vector<vector<size_t>> getInputShapes() {
            return inputShapes;
        }

        vector<size_t> getOutputShape() {
            return outputShape;
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            weights->save(path);
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            this->weights = make_shared<FullTensor>(path);
        }

        // predicting
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw runtime_error("FullyConnectedNeurons only supports a single input.");
            }

            auto lastInput = input[0];
            if (forTraining) {
                lastInputs.push(lastInput);
            }

            return make_shared<MatrixMultiplyTensorView>(lastInput, weights);
        }

        // learning
        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw runtime_error("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            shared_ptr<BaseTensor> average_last_inputs = lastInputs.front();
            lastInputs.pop();
            while (!lastInputs.empty()) {
                auto nextLastInput = lastInputs.front();
                lastInputs.pop();
                average_last_inputs = make_shared<AddTensorView>(average_last_inputs, nextLastInput);
            }
            if (lastInputsSize > 1) {
                average_last_inputs = materializeTensor(
                        make_shared<ScalarMultiplyTensorView>(average_last_inputs, 1.f / (float) lastInputsSize));
            }

            // find the error
            auto weights_transposed = make_shared<TransposeTensorView>(weights);
            // TODO: we greatly improve performance by materializing the tensor into a FullTensor here, but sometimes this will use
            //  considerably more memory than we need. Part of me thinks that all dot product tensors should be materialized,
            //  and part of me thinks that there are situations of simple dot products don't need to be.
            shared_ptr<BaseTensor> input_error = make_shared<FullTensor>(
                    make_shared<MatrixMultiplyTensorView>(output_error, weights_transposed));

            // update weights
            auto input_transposed = make_shared<TransposeTensorView>(average_last_inputs);
            auto weights_error = make_shared<MatrixMultiplyTensorView>(input_transposed, output_error);

            const auto adjusted_weights_error = make_shared<ScalarMultiplyTensorView>(weights_error,
                                                                                      mixedPrecisionScale);
            const auto adjusted_weights = optimizer->calculateWeightsChange(
                    registration_id, weights, adjusted_weights_error);

            weights = materializeTensor(adjusted_weights, bits);

            return {input_error};
        }

    private:
        shared_ptr<BaseTensor> weights;
        int registration_id;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
        uint8_t bits;
        float mixedPrecisionScale;
        vector<vector<size_t>> inputShapes;
        vector<size_t> outputShape;
        shared_ptr<BaseOptimizer> optimizer;
        string label;
    };
}
#endif //HAPPYML_FULLY_CONNECTED_LAYER_HPP
