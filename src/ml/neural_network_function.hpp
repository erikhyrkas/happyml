//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_NEURAL_NETWORK_FUNCTION_HPP
#define HAPPYML_NEURAL_NETWORK_FUNCTION_HPP

#include "activation.hpp"
#include "optimizer.hpp"
#include "../util/tensor_utils.hpp"
#include "../util/basic_profiler.hpp"

namespace happyml {
    // side note: I read an article on back propagation I thought was interesting:
    // https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d

    class NeuralNetworkFunction {
    public:
        virtual shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) = 0;

        virtual shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) = 0;

        virtual void saveKnowledge(const string &fullKnowledgePath) {
        }

        virtual void loadKnowledge(const string &fullKnowledgePath) {
        }
    };

    class NeuralNetworkActivationFunction : public NeuralNetworkFunction {
    public:
        explicit NeuralNetworkActivationFunction(const shared_ptr<ActivationFunction> &activationFunction) {
            this->activationFunction = activationFunction;
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            // todo: throw error on wrong size input?
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw exception("Cannot activate multiple inputs at the same time. Please merge.");
            }
            const auto &lastInput = input[0];
            if (forTraining) {
                lastInputs.push(lastInput);
            }
            return activationFunction->activate(lastInput);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &outputError) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw exception("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            // TODO: it's really inefficient to calculate the derivative of every previous batch input and average it
            //  but doing an average first and then a derivative isn't right.
            //  I think I'm doing the back propagation incorrectly for mini-batch here.
            shared_ptr<BaseTensor> averageActivationDerivative = activationFunction->derivative(lastInputs.front());
            lastInputs.pop();
            while (!lastInputs.empty()) {
                auto nextLastInput = activationFunction->derivative(lastInputs.front());
                lastInputs.pop();
                averageActivationDerivative = make_shared<TensorAddTensorView>(averageActivationDerivative,
                                                                               nextLastInput);
            }
            if (lastInputsSize > 1) {
                averageActivationDerivative = materializeTensor(
                        make_shared<TensorMultiplyByScalarView>(averageActivationDerivative,
                                                                1.f / (float) lastInputsSize));
            }

            //auto activation_derivative = activationFunction->derivative(average_last_inputs);
            // this really threw me for a loop. I thought that this was supposed to be dot product, rather than
            // an element-wise-multiplication.
            const auto baseOutputError = make_shared<TensorMultiplyTensorView>(averageActivationDerivative,
                                                                               outputError);
            return baseOutputError;
        }

    private:
        shared_ptr<ActivationFunction> activationFunction;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
    };

    class NeuralNetworkFlattenFunction : public NeuralNetworkFunction {
    public:
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw exception("Cannot flatten multiple inputs at the same time. Please merge.");
            }
            const auto &nextInput = input[0];
            originalCols = nextInput->columnCount();
            originalRows = nextInput->rowCount();
            if (originalRows == 1) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return nextInput;
            }
            return make_shared<TensorFlattenToRowView>(nextInput);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            if (originalRows == output_error->rowCount() && originalCols == output_error->columnCount()) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return output_error;
            }
            return make_shared<TensorReshapeView>(output_error, originalRows, originalCols);
        }

    private:
        size_t originalRows{};
        size_t originalCols{};
    };


    // Here's an interesting, related read:
    // https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
    // also:
    // https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class Convolution2dValidFunction : public NeuralNetworkFunction {
    public:
        Convolution2dValidFunction(const string &label,
                                   vector<size_t> inputShape, size_t filters, size_t kernelSize, uint8_t bits,
                                   const shared_ptr<BaseOptimizer> &optimizer) {
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

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw exception("Convolution2dValidFunction only supports a single input.");
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
                    const auto weightForInputLayer = make_shared<TensorChannelToTensorView>(weights[outputLayer],
                                                                                            inputLayer);
                    const auto inputChannel = make_shared<TensorChannelToTensorView>(lastInput, inputLayer);
                    const auto correlation2d = make_shared<TensorValidCrossCorrelation2dView>(inputChannel,
                                                                                              weightForInputLayer);
                    if (outputTensor) {
                        outputTensor = make_shared<TensorAddTensorView>(outputTensor, correlation2d);
                    } else {
                        outputTensor = correlation2d;
                    }
                }
                const shared_ptr<BaseTensor> summedCorrelation2d = make_shared<TensorSumToChannelView>(outputTensor,
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

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &outputError) override {
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw exception("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            shared_ptr<BaseTensor> averageLastInputs = lastInputs.front();
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
            shared_ptr<BaseTensor> inputError = nullptr;
            for (size_t outputLayer = 0; outputLayer < filters; outputLayer++) {
                const auto outputErrorForLayer = make_shared<TensorChannelToTensorView>(outputError, outputLayer);
                shared_ptr<BaseTensor> weightChanges = nullptr;
                for (size_t inputLayer = 0; inputLayer < inputDepth; inputLayer++) {
                    const auto weightForInputLayer = make_shared<TensorChannelToTensorView>(weights[outputLayer],
                                                                                            inputLayer);
                    const auto nextInputError = make_shared<TensorFullConvolve2dView>(outputErrorForLayer,
                                                                                      weightForInputLayer);
                    const auto inputErrorToInputChannel = make_shared<TensorSumToChannelView>(nextInputError,
                                                                                              inputLayer, inputDepth);
                    if (inputError) {
                        inputError = make_shared<TensorAddTensorView>(inputError, inputErrorToInputChannel);
                    } else {
                        inputError = inputErrorToInputChannel;
                    }
                    const auto inputLayerChannel = make_shared<TensorChannelToTensorView>(averageLastInputs,
                                                                                          inputLayer);
                    const auto nextWeightError = make_shared<TensorValidCrossCorrelation2dView>(inputLayerChannel,
                                                                                                outputErrorForLayer);
                    const auto nextWeightToInputChannel = make_shared<TensorSumToChannelView>(nextWeightError,
                                                                                              inputLayer, inputDepth);
                    if (weightChanges) {
                        weightChanges = make_shared<TensorAddTensorView>(weightChanges, nextWeightToInputChannel);
                    } else {
                        weightChanges = nextWeightToInputChannel;
                    }
                }

                const auto adjustedWeights = optimizer->calculateWeightsChange(registration_id,
                                                                               weights[outputLayer],
                                                                               weightChanges,
                                                                               mixedPrecisionScale);
                weights[outputLayer] = materializeTensor(adjustedWeights, bits);
            }

            const auto resultError = make_shared<TensorSumChannelsView>(inputError);
            return resultError;
        }

    private:
        int registration_id;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
        vector<shared_ptr<BaseTensor>> weights;
        uint8_t bits;
        float mixedPrecisionScale;
        vector<size_t> inputShape;
        vector<size_t> outputShape;
        size_t kernelSize;
        shared_ptr<BaseOptimizer> optimizer;
        string label;
    };

    class FullyConnectedNeurons : public NeuralNetworkFunction {
    public:
        FullyConnectedNeurons(const string &label, size_t inputSize, size_t outputSize, uint8_t bits,
                              const shared_ptr<BaseOptimizer> &optimizer) {
            this->label = label;
            this->registration_id = optimizer->registerForWeightChanges();
            this->inputShapes = vector<vector<size_t >>{{1, inputSize, 1}};
            this->outputShape = vector<size_t>{1, outputSize, 1};
            this->weights = make_shared<TensorFromRandom>(inputSize, outputSize, 1, -0.5f, 0.5f, 42);
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
                throw exception("FullyConnectedNeurons only supports a single input.");
            }

            auto lastInput = input[0];
            if (forTraining) {
                lastInputs.push(lastInput);
            }

            return make_shared<TensorMatrixMultiplyTensorView>(lastInput, weights);
        }

        // learning
        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw exception("FullyConnectedNeurons.backward() called without previous inputs.");
            }
            shared_ptr<BaseTensor> average_last_inputs = lastInputs.front();
            lastInputs.pop();
            while (!lastInputs.empty()) {
                auto nextLastInput = lastInputs.front();
                lastInputs.pop();
                average_last_inputs = make_shared<TensorAddTensorView>(average_last_inputs, nextLastInput);
            }
            if (lastInputsSize > 1) {
                average_last_inputs = materializeTensor(
                        make_shared<TensorMultiplyByScalarView>(average_last_inputs, 1.f / (float) lastInputsSize));
            }

            // find the error
            auto weights_transposed = make_shared<TensorTransposeView>(weights);
            // TODO: we greatly improve performance by materializing the tensor into a FullTensor here, but sometimes this will use
            //  considerably more memory than we need. Part of me thinks that all dot product tensors should be materialized,
            //  and part of me thinks that there are situations of simple dot products don't need to be.
            shared_ptr<BaseTensor> input_error = make_shared<FullTensor>(
                    make_shared<TensorMatrixMultiplyTensorView>(output_error, weights_transposed));

            // update weights
            auto input_transposed = make_shared<TensorTransposeView>(average_last_inputs);
            auto weights_error = make_shared<TensorMatrixMultiplyTensorView>(input_transposed, output_error);

            const auto adjusted_weights = optimizer->calculateWeightsChange(
                    registration_id, weights, weights_error, mixedPrecisionScale);

            weights = materializeTensor(adjusted_weights, bits);

            return input_error;
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

    class BiasNeuron : public NeuralNetworkFunction {
    public:
        BiasNeuron(const string &label, const vector<size_t> &inputShape, const vector<size_t> &outputShape,
                   uint8_t bits,
                   const shared_ptr<BaseOptimizer> &optimizer) {
            this->label = label;
            this->registration_id = optimizer->registerForBiasChanges();
            this->inputShapes = vector<vector<size_t >>{inputShape};
            this->outputShape = outputShape;
            // In my experiments, at least for the model I was testing, we found the correct results faster by starting at 0 bias.
            // This may be a mistake.
            this->bias = make_shared<UniformTensor>(outputShape[0], outputShape[1], outputShape[2], 0.f);
            // Original code started with a random value between -0.5 and 0.5:
            //this->bias = make_shared<TensorFromRandom>(outputShape[0], outputShape[1],outputShape[2], -0.5f, 0.5f, 42);
            this->bits = bits;
            this->optimizer = optimizer;
            // With models that are not fully 32-bit, if you don't scale the loss
            // you'll have precision errors that are difficult to deal with.
            // I chose to scale the learning rate, which brings a lot of potential issues
            // but is relatively straight forward to do and fast.
            // The biggest issue is that the caller might try to use a learning rate that is too big,
            // and it will not be possible to find good results.
            // There is an nvidia paper on the topic, and they scale the values before they store the weights
            // and then scale those weights back down when they use them. The advantage of this is that
            // a learning rate of X on a 32-bit model, it will work the same if you change some portions
            // to 16-bit. With my approach, if you change any of the portions of the model's precision, you may
            // have to pick new a new learning rate to get good results.
            // I went this route because it is expensive to scale up and down tensors using a view with
            // this framework when you end up with a huge stack of views, since that scaling will constantly
            // get re-applied with every future view that sits over weights.
            // There are situations where I have hundreds of views over a tensor, and adding a single view to
            // the weights will change that to thousands because many of the views sit over multiple weight
            // tensors.
            if (bits == 32) {
                // NOTE: I am taking a small shortcut here. Even without mixed precision, it's important to
                //  reduce the rate we train bias. If bias is trained at the same rate as weights, my observation
                //  is that it can "overpower" the weights, where it causes us to wildly oscillate above and below
                //  our target without ever reaching it.
                // I made this number up. it seemed to work well for both mixed-precision models and for models
                // that are entirely 32-bit.
                mixedPrecisionScale = 0.1f;
            } else if (bits == 16) {
                if (optimizer->getLearningRate() < 0.45) {
                    // I made this number up. it seemed to work well for mixed-precision models.
                    mixedPrecisionScale = 2.f;
                } else {
                    mixedPrecisionScale = 1.f;
                }
            } else {
                if (optimizer->getLearningRate() < 0.3) {
                    // I made this number up. it seemed to work well for mixed-precision models.
                    mixedPrecisionScale = 3.0f;
                } else {
                    mixedPrecisionScale = 1.0f;
                }
            }
            this->current_batch_size = 0;
        }

        vector<vector<size_t>> getInputShapes() {
            return inputShapes;
        }

        vector<size_t> getOutputShape() {
            return outputShape;
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            bias->save(path);
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            this->bias = make_shared<FullTensor>(path);
        }

        // predicting
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw exception("BiasNeuron only supports a single input.");
            }
            if (forTraining) {
                current_batch_size++;
            }

            return make_shared<TensorAddTensorView>(input[0], bias);
        }

        // learning
        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);

            auto adjusted_bias = optimizer->calculateBiasChange(registration_id,
                                                                bias,
                                                                output_error,
                                                                mixedPrecisionScale,
                                                                (float) current_batch_size);
            bias = materializeTensor(adjusted_bias, bits);

            current_batch_size = 0;
            // TODO: partial derivative of bias would always be 1, so we pass along original error. I'm fairly sure this is right.
            // but I notice that the quarter float doesn't handle big shifts in scale very well
            return output_error;
        }

    private:
        int registration_id;
        shared_ptr<BaseTensor> bias;
        int current_batch_size;
        uint8_t bits;
        float mixedPrecisionScale;
        vector<vector<size_t>> inputShapes;
        vector<size_t> outputShape;
        shared_ptr<BaseOptimizer> optimizer;
        string label;
    };

}
#endif //HAPPYML_NEURAL_NETWORK_FUNCTION_HPP
