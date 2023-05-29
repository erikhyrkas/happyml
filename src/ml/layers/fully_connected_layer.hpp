//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FULLY_CONNECTED_LAYER_HPP
#define HAPPYML_FULLY_CONNECTED_LAYER_HPP

#include "../base_layer.hpp"
#include "../../types/tensor_impls/tensor_from_xavier.hpp"
#include "../../types/tensor_views/standardize_tensor_view.hpp"
#include "../../types/tensor_views/standardize_derivative_tensor_view.hpp"

namespace happyml {
    class FullyConnectedLayer : public BaseLayer {
    public:
        FullyConnectedLayer(const string &label, size_t inputSize, size_t outputSize, uint8_t bits,
                            int optimizer_registration_id,
                            bool use_l2_regularization = true) : weights_errors() {
            this->label = label;
            this->registration_id = optimizer_registration_id;
            this->inputShapes = vector<vector<size_t >>{{1, inputSize, 1}};
            this->outputShape = vector<size_t>{1, outputSize, 1};
            this->weights = make_shared<TensorFromXavier>(inputSize, outputSize, 1, optimizer_registration_id + 42);
            this->bits = bits;
            this->use_l2_regularization = use_l2_regularization;
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

            shared_ptr<BaseTensor> result = make_shared<MatrixMultiplyTensorView>(lastInput, weights);

#ifdef DEBUG_TRAIN_NAN
            if (result->hasNaNOrInf()) {
                throw runtime_error("FullyConnectedNeurons.forward() result has NaN or Inf");
            }
#endif
            return result;
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
            if (lastInputsSize > 1) {
                PROFILE_BLOCK(multiInputProfileBlock);
                while (!lastInputs.empty()) {
                    auto nextLastInput = lastInputs.front();
                    lastInputs.pop();
                    average_last_inputs = make_shared<AddTensorView>(average_last_inputs, nextLastInput);
                }
                average_last_inputs = materializeTensor(
                        make_shared<ScalarDivideTensorView>(average_last_inputs, (float) lastInputsSize));
            }

            // find the error
            auto weights_transposed = make_shared<TransposeTensorView>(weights);
            // TODO: we greatly improve performance by materializing the tensor into a FullTensor here, but sometimes this will use
            //  considerably more memory than we need. Part of me thinks that all dot product tensors should be materialized,
            //  and part of me thinks that there are situations of simple dot products don't need to be.
            shared_ptr<BaseTensor> input_error = materializeTensor(
                    make_shared<MatrixMultiplyTensorView>(output_error, weights_transposed));

            // update weights
            auto input_transposed = make_shared<TransposeTensorView>(average_last_inputs);

            shared_ptr<BaseTensor> weights_error = make_shared<MatrixMultiplyTensorView>(input_transposed, output_error);

            if (use_l2_regularization) {
                PROFILE_BLOCK(profileBlock2);

                // Calculate L2 regularization term
                auto l2_regularization = make_shared<ScalarMultiplyTensorView>(weights, regularization_param);

                // Add L2 regularization term to weights error
                weights_error = make_shared<AddTensorView>(weights_error, l2_regularization);
#ifdef DEBUG_TRAIN_NAN
                if (weights->hasNaNOrInf()) {
                    throw runtime_error("FullyConnectedNeurons.backward() weights has NaN or Inf");
                }
                if (weights_squared->hasNaNOrInf()) {
                    weights->print();
                    weights_squared->print();
                    throw runtime_error("FullyConnectedNeurons.backward() weights_squared has NaN or Inf");
                }
                if (l2_regularization->hasNaNOrInf()) {
                    throw runtime_error("FullyConnectedNeurons.backward() l2_regularization has NaN or Inf");
                }
                if (weights_error->hasNaNOrInf()) {
                    throw runtime_error("FullyConnectedNeurons.backward() weights_error has NaN or Inf");
                }
#endif
            }
            weights_errors.push(weights_error);

            return {input_error};
        }

        void apply(const shared_ptr<BaseOptimizer> &optimizer) override {
            PROFILE_BLOCK(profileBlock);

            if (weights_errors.empty()) {
                throw runtime_error("FullyConnectedNeurons.apply() called without previous weights_errors.");
            }

            size_t weights_errors_size = weights_errors.size();
            auto weights_error = weights_errors.front();
            weights_errors.pop();
            while (!weights_errors.empty()) {
                auto next_weights_error = weights_errors.front();
                weights_errors.pop();
                weights_error = make_shared<AddTensorView>(weights_error, next_weights_error);
            }
            weights_error = materializeTensor(
                    make_shared<ScalarDivideTensorView>(weights_error,
                                                        (float) weights_errors_size));

            const auto adjusted_weights = optimizer->calculateWeightsChange(
                    registration_id, weights, weights_error);

            weights = materializeTensor(adjusted_weights, bits);
        }

        bool is_trainable() override {
            return true;
        }

        size_t get_parameter_count() override {
            return weights->size();
        }

    private:
        shared_ptr<BaseTensor> weights;
        int registration_id;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
        uint8_t bits;
        vector<vector<size_t>> inputShapes;
        vector<size_t> outputShape;
        queue<shared_ptr<BaseTensor>> weights_errors;
        bool use_l2_regularization;
        const float regularization_param = 0.02f; // sane default of 2 * 0.01f
        string label;
    };

    // TODO: make layers thread safe. For example:
    // #include <concurrentqueue/concurrentqueue.h>
    //
    //concurrentqueue::concurrent_queue<shared_ptr<StandardizeTensorView>> lastInputs;
    //
    //void push(shared_ptr<StandardizeTensorView> tensor) {
    //    lastInputs.push(tensor);
    //}
    //
    //shared_ptr<StandardizeTensorView> pop() {
    //    return lastInputs.pop();
    //}
}
#endif //HAPPYML_FULLY_CONNECTED_LAYER_HPP
