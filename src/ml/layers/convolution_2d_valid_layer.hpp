//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP
#define HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP

namespace happyml {
// Here's an interesting, related read:
// https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
// also:
// https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class Convolution2dValidFunction : public happyml::NeuralNetworkLayerFunction {
    public:
        Convolution2dValidFunction(const string &label,
                                   vector<size_t> inputShape, size_t filters, size_t kernelSize, uint8_t bits,
                                   const shared_ptr<happyml::BaseOptimizer> &optimizer) {
            this->label = label;
            this->registration_id = optimizer->registerForWeightChanges();
            this->inputShape = inputShape;
            this->kernelSize = kernelSize;
            this->outputShape = {inputShape[0] - kernelSize + 1, inputShape[1] - kernelSize + 1, filters};
            this->bits = bits;
            this->weights = {};
            for (size_t next_weight_layer = 0; next_weight_layer < filters; next_weight_layer++) {
                this->weights.push_back(
                        make_shared<TensorFromRandom>(kernelSize, kernelSize, inputShape[2], -0.5f, 0.5f, 42));
            }
            this->optimizer = optimizer;
            if (bits == 32) {
                mixedPrecisionScale = 0.5f;
            } else if (bits == 16) {
                mixedPrecisionScale = 2.f;
            } else {
                mixedPrecisionScale = 3.f;
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

        shared_ptr<happyml::BaseTensor> forward(const vector<shared_ptr<happyml::BaseTensor>> &input, bool forTraining) override {
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
            shared_ptr<happyml::BaseTensor> result = nullptr;
            for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                shared_ptr<happyml::BaseTensor> outputTensor = nullptr;
                for (size_t inputLayer = 0; inputLayer < inputDepth; inputLayer++) {
                    const auto weightForInputLayer = make_shared<happyml::TensorChannelToTensorView>(weights[outputLayer],
                                                                                                     inputLayer);
                    const auto inputChannel = make_shared<happyml::TensorChannelToTensorView>(lastInput, inputLayer);
                    const auto correlation2d = make_shared<happyml::TensorValidCrossCorrelation2dView>(inputChannel,
                                                                                                       weightForInputLayer);
                    if (outputTensor) {
                        outputTensor = make_shared<TensorAddTensorView>(outputTensor, correlation2d);
                    } else {
                        outputTensor = correlation2d;
                    }
                }
                const shared_ptr<happyml::BaseTensor> summedCorrelation2d = make_shared<happyml::TensorSumToChannelView>(outputTensor,
                                                                                                                         outputLayer,
                                                                                                                         filters);
                if (!result) {
                    result = summedCorrelation2d;
                } else {
                    // each summed correlation 2d tensor is in its own output channel
                    result = make_shared<TensorAddTensorView>(result, summedCorrelation2d);
                }
            }
            // todo: it would be faster to have some sort of CombinedTensor where rather than adding the tensors,
            //  passed a vector of tensors and we only use 1 layer from each. The tensor add object will
            //  cause us to add a 0 to each value for each layer. If you have tensors added together, you'll
            //  have an 256 additional + operations for every value you fetch -- and those operations aren't
            //  changing the outcome.

            return result;
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<happyml::BaseTensor> &outputError) override {
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw runtime_error("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            shared_ptr<happyml::BaseTensor> averageLastInputs = lastInputs.front();
            lastInputs.pop();
            while (!lastInputs.empty()) {
                auto nextLastInput = lastInputs.front();
                lastInputs.pop();
                averageLastInputs = make_shared<TensorAddTensorView>(averageLastInputs, nextLastInput);
            }
            if (lastInputsSize > 1) {
                averageLastInputs = materializeTensor(
                        make_shared<TensorMultiplyByScalarView>(averageLastInputs, 1.f / (float) lastInputsSize));
            }

            // input error for each input channel is
            // the sum of the fullConvolve2d of the output errors and the weights
            // filters are the number of output channels we have
            const size_t filters = outputShape[2];
            const size_t inputDepth = inputShape[2];
            shared_ptr<happyml::BaseTensor> inputError = nullptr;
            for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                const auto outputErrorForLayer = make_shared<happyml::TensorChannelToTensorView>(outputError, outputLayer);
                shared_ptr<happyml::BaseTensor> weightChanges = nullptr;
                for (size_t inputLayer = 0; inputLayer < inputDepth; inputLayer++) {
                    const auto weightForInputLayer = make_shared<happyml::TensorChannelToTensorView>(weights[outputLayer],
                                                                                                     inputLayer);
                    const auto nextInputError = make_shared<happyml::TensorFullConvolve2dView>(outputErrorForLayer,
                                                                                               weightForInputLayer);
                    const auto inputErrorToInputChannel = make_shared<happyml::TensorSumToChannelView>(nextInputError,
                                                                                                       inputLayer, inputDepth);
                    if (inputError) {
                        inputError = make_shared<TensorAddTensorView>(inputError, inputErrorToInputChannel);
                    } else {
                        inputError = inputErrorToInputChannel;
                    }
                    const auto inputLayerChannel = make_shared<happyml::TensorChannelToTensorView>(averageLastInputs,
                                                                                                   inputLayer);
                    const auto nextWeightError = make_shared<happyml::TensorValidCrossCorrelation2dView>(inputLayerChannel,
                                                                                                         outputErrorForLayer);
                    const auto nextWeightToInputChannel = make_shared<happyml::TensorSumToChannelView>(nextWeightError,
                                                                                                       inputLayer, inputDepth);
                    if (weightChanges) {
                        weightChanges = make_shared<TensorAddTensorView>(weightChanges, nextWeightToInputChannel);
                    } else {
                        weightChanges = nextWeightToInputChannel;
                    }
                }

                const auto adjusted_weights_error = make_shared<TensorMultiplyByScalarView>(weightChanges,
                                                                                            mixedPrecisionScale);
                const auto adjustedWeights = optimizer->calculateWeightsChange(registration_id,
                                                                               weights[outputLayer],
                                                                               adjusted_weights_error);
                weights[outputLayer] = materializeTensor(adjustedWeights, bits);
            }

            const auto resultError = make_shared<happyml::TensorSumChannelsView>(inputError);
            return {resultError};
        }

    private:
        int registration_id;
        queue<shared_ptr<happyml::BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
        vector<shared_ptr<happyml::BaseTensor>> weights;
        uint8_t bits;
        float mixedPrecisionScale;
        vector<size_t> inputShape;
        vector<size_t> outputShape;
        size_t kernelSize;
        shared_ptr<happyml::BaseOptimizer> optimizer;
        string label;
    };
}

#endif //HAPPYML_CONVOLUTION_2D_VALID_LAYER_HPP
