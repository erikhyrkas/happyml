//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP
#define HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP

#include "../../types/tensor_impls/tensor_from_xavier.hpp"

namespace happyml {
// Here's an interesting, related read:
// https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
// also:
// https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class Convolution2dValidFunction : public BaseLayer {
    public:
        Convolution2dValidFunction(const string &label,
                                   vector<size_t> inputShape, size_t filters, size_t kernelSize, uint8_t bits,
                                   int optimizer_registration_id,
                                   bool use_l2_regularization,
                                   float regularization_strength) {
            this->label = label;
            this->registration_id = optimizer_registration_id;
            this->inputShape = inputShape;
            this->kernelSize = kernelSize;
            this->outputShape = {inputShape[0] - kernelSize + 1, inputShape[1] - kernelSize + 1, filters};
            this->use_l2_regularization = use_l2_regularization;
            this->bits = bits;
            this->weights = {};
            this->regularization_strength = regularization_strength;
            for (size_t next_weight_layer = 0; next_weight_layer < filters; next_weight_layer++) {
                this->weights.push_back(make_shared<TensorFromXavier>(kernelSize, kernelSize, inputShape[2], optimizer_registration_id + next_weight_layer + 42));
            }
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            auto filters = outputShape[2];
            for (size_t next_weight_layer = 0; next_weight_layer < filters; next_weight_layer++) {
                string path = fullKnowledgePath + "/" + label + "_" + asString(next_weight_layer) + ".tensor";
                weights[next_weight_layer]->save(path);
            }
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            this->weights = {};
            auto filters = outputShape[2];
            for (size_t next_weight_layer = 0; next_weight_layer < filters; next_weight_layer++) {
                string path = fullKnowledgePath + "/" + label + "_" + asString(next_weight_layer) + ".tensor";
                auto matrix = make_shared<FullTensor>(path);
                this->weights.push_back(matrix);
            }
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw runtime_error("Convolution2dValidFunction only supports a single input.");
            }

            auto lastInput = input[0];
            if (forTraining) {
                lastInputs.push(lastInput);
            }

            // filters are the number of output channels we have
            const size_t filters = outputShape[2];
            const size_t inputDepth = inputShape[2];
            shared_ptr<BaseTensor> result = nullptr;
            for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                shared_ptr<BaseTensor> outputTensor = nullptr;
                for (size_t inputLayer = 0; inputLayer < inputDepth; inputLayer++) {
                    const auto weightForInputLayer = make_shared<ChannelToTensorView>(weights[outputLayer],
                                                                                      inputLayer);
                    const auto inputChannel = make_shared<ChannelToTensorView>(lastInput, inputLayer);
                    const auto correlation2d = make_shared<Valid2DCrossCorrelationTensorView>(inputChannel,
                                                                                              weightForInputLayer);
                    if (outputTensor) {
                        outputTensor = make_shared<AddTensorView>(outputTensor, correlation2d);
                    } else {
                        outputTensor = correlation2d;
                    }
                }
                const shared_ptr<BaseTensor> summedCorrelation2d = make_shared<SumToChannelTensorView>(outputTensor,
                                                                                                       outputLayer,
                                                                                                       filters);
                if (!result) {
                    result = summedCorrelation2d;
                } else {
                    // each summed correlation 2d tensor is in its own output channel
                    result = make_shared<AddTensorView>(result, summedCorrelation2d);
                }
            }
            // todo: it would be faster to have some sort of CombinedTensor where rather than adding the tensors,
            //  passed a vector of tensors and we only use 1 layer from each. The tensor add object will
            //  cause us to add a 0 to each value for each layer. If you have tensors added together, you'll
            //  have an 256 additional + operations for every value you fetch -- and those operations aren't
            //  changing the outcome.

            return result;
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &outputError) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw runtime_error("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            shared_ptr<BaseTensor> averageLastInputs = lastInputs.front();
            lastInputs.pop();
            if (lastInputsSize > 1) {
                while (!lastInputs.empty()) {
                    auto nextLastInput = lastInputs.front();
                    lastInputs.pop();
                    averageLastInputs = make_shared<AddTensorView>(averageLastInputs, nextLastInput);
                }
                averageLastInputs = materializeTensor(
                        make_shared<ScalarMultiplyTensorView>(averageLastInputs, 1.f / (float) lastInputsSize));
            }

            // input error for each input channel is
            // the sum of the fullConvolve2d of the output errors and the weights
            // filters are the number of output channels we have
            const size_t filters = outputShape[2];
            const size_t inputDepth = inputShape[2];
            shared_ptr<BaseTensor> inputError = make_shared<UniformTensor>(inputShape, 0.0f);
            vector<shared_ptr<BaseTensor>> output_layers_weight_changes;
            for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                const auto outputErrorForLayer = make_shared<ChannelToTensorView>(outputError, outputLayer);
                shared_ptr<BaseTensor> output_weight_changes = make_shared<UniformTensor>(weights[outputLayer]->getShape(), 0.0f);;
                for (size_t inputLayer = 0; inputLayer < inputDepth; inputLayer++) {
                    const auto weightForInputLayer = make_shared<ChannelToTensorView>(weights[outputLayer],
                                                                                      inputLayer);
                    const auto nextInputError = make_shared<Full2DConvolveTensorView>(outputErrorForLayer,
                                                                                      weightForInputLayer);
                    const auto inputErrorToInputChannel = make_shared<SumToChannelTensorView>(nextInputError,
                                                                                              inputLayer,
                                                                                              inputDepth);
                    inputError = make_shared<AddTensorView>(inputError, inputErrorToInputChannel);
                    const auto inputLayerChannel = make_shared<ChannelToTensorView>(averageLastInputs,
                                                                                    inputLayer);
                    shared_ptr<BaseTensor> nextWeightError = make_shared<Valid2DCrossCorrelationTensorView>(inputLayerChannel,
                                                                                                            outputErrorForLayer);
                    if (use_l2_regularization) {
                        PROFILE_BLOCK(profileBlock2);
                        auto l2_regularization = make_shared<ScalarMultiplyTensorView>(nextWeightError, regularization_strength);
                        nextWeightError = make_shared<AddTensorView>(nextWeightError, l2_regularization);
                    }
                    const auto nextWeightToInputChannel = make_shared<SumToChannelTensorView>(nextWeightError,
                                                                                              inputLayer, inputDepth);

                    output_weight_changes = make_shared<AddTensorView>(output_weight_changes, nextWeightToInputChannel);
                }
                output_layers_weight_changes.push_back(output_weight_changes);
            }

            weight_changes.push(output_layers_weight_changes);

            const auto resultError = make_shared<SumChannelsTensorView>(inputError);
            return {resultError};
        }

        void apply(const shared_ptr<BaseOptimizer> &optimizer) override {
            PROFILE_BLOCK(profileBlock);

            const size_t filters = outputShape[2];
            while (!weight_changes.empty()) {
                vector<shared_ptr<BaseTensor>> weightChanges = weight_changes.front();
                weight_changes.pop();
                for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                    auto loss_gradient = weightChanges[outputLayer];
                    const auto adjustedWeights = optimizer->calculateWeightsChange(registration_id,
                                                                                   weights[outputLayer],
                                                                                   loss_gradient);
                    weights[outputLayer] = materializeTensor(adjustedWeights, bits);
                }
            }
        }

        bool is_trainable() override {
            return true;
        }

        size_t get_parameter_count() override {
            size_t total = 0;
            for (const auto &weight: weights) {
                total += weight->size();
            }
            return total;
        }

        [[nodiscard]] size_t get_kernel_size() const {
            return kernelSize;
        }

    private:
        int registration_id;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
        queue<vector<shared_ptr<BaseTensor>>> weight_changes;
        vector<shared_ptr<BaseTensor>> weights;
        uint8_t bits;
        vector<size_t> inputShape;
        vector<size_t> outputShape;
        size_t kernelSize;
        string label;
        bool use_l2_regularization;
        float regularization_strength;
    };
}

#endif //HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP
