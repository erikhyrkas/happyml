//
// Created by Erik Hyrkas on 11/2/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_HAPPYML_DSL_HPP
#define HAPPYML_HAPPYML_DSL_HPP

#include <iostream>
#include <chrono>
#include <utility>
#include "enums.hpp"
#include "optimizer_factory.hpp"
#include "neural_network_node.hpp"
#include "activators/leaky_relu_activation_function.hpp"
#include "activators/relu_activation_function.hpp"
#include "activators/linear_activation_function.hpp"
#include "activators/tanh_activation_function.hpp"
#include "activators/tanh_approx_activation_function.hpp"
#include "activators/softmax_activation_function.hpp"
#include "activators/sigmoid_approx_activation_function.hpp"
#include "activators/sigmoid_activation_function.hpp"
#include "neural_network.hpp"
#include "layers/activation_layer.hpp"
#include "layers/flatten_layer.hpp"
#include "layers/normalization_layer.hpp"
#include "layers/convolution_2d_valid_layer.hpp"
#include "layers/concatenate_wide_layer.hpp"
#include "layers/fully_connected_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/bias_layer.hpp"

using namespace happyml;
using namespace std;

namespace happyml {

    class HappymlDSL : public enable_shared_from_this<HappymlDSL> {
    public:
        explicit HappymlDSL(OptimizerType optimizerType, const string &modelName = "unnamed",
                            const string &repoRootPath = "repo") {
            this->optimizerType = optimizerType;
            this->modelName = modelName;
            if (!std::all_of(modelName.begin(), modelName.end(),
                             [](int c) { return std::isalnum(c) || c == '_'; })) {
                throw runtime_error("Model name must contain only alphanumeric characters.");
            }
            switch (optimizerType) {
                case sgd:
                    this->learningRate = 0.005;
                    this->biasLearningRate = 0.001;
                    break;
                default:
                    this->learningRate = 0.001;
                    this->biasLearningRate = 0.001;
            }
            this->lossType = LossType::mse;
            this->repoRootPath = repoRootPath;
            this->vertexUniqueSequenceCounter = 0;
        }

        shared_ptr<HappymlDSL> setBiasLearningRate(float biasLearningRateValue) {
            this->biasLearningRate = biasLearningRateValue;
            return shared_from_this();
        }

        shared_ptr<HappymlDSL> setLearningRate(float learningRateValue) {
            this->learningRate = learningRateValue;
            return shared_from_this();
        }

        shared_ptr<HappymlDSL> setLossFunction(LossType lossTypeValue) {
            this->lossType = lossTypeValue;
            return shared_from_this();
        }

        shared_ptr<HappymlDSL> setModelName(const string &modelNameValue) {
            this->modelName = modelNameValue;
            if (!std::all_of(modelName.begin(), modelName.end(),
                             [](int c) { return std::isalnum(c) || c == '_'; })) {
                throw runtime_error("Model name must contain only alphanumeric characters.");
            }
            return shared_from_this();
        }

        shared_ptr<HappymlDSL> setModelRepo(const string &modelRepoPath) {
            this->repoRootPath = modelRepoPath;
            return shared_from_this();
        }

        shared_ptr<NeuralNetworkForTraining> build() {


            auto neuralNetwork = make_shared<NeuralNetworkForTraining>(this->modelName,
                                                                       repoRootPath,
                                                                       optimizerType,
                                                                       learningRate,
                                                                       biasLearningRate,
                                                                       lossType);
            vector<vector<string>> networkMetadata;
            networkMetadata.push_back({"optimizer", optimizerTypeToString(optimizerType)});
            networkMetadata.push_back({"learningRate", asString(learningRate)});
            networkMetadata.push_back({"biasLearningRate", asString(biasLearningRate)});
            networkMetadata.push_back({"loss", lossTypeToString(lossType)});
            // buildNode will add two types of metadata
            // first it will add a vertex record:
            // "vertex", id, is input, is output, node type, activation type, materialized, uses bias, bits,
            // input rows, input columns, input channels, output rows, output columns, output channels, filters, kernels

            // and then it will add any edge records:
            // "edge", from id, to id, to id, to id...

            for (const auto &head: inputReceptors) {
                neuralNetwork->addHeadNode(head->buildLayer(neuralNetwork, networkMetadata));
            }

            neuralNetwork->setNetworkMetadata(networkMetadata);
            return neuralNetwork;
        }

        uint32_t nextVertexId() {
            // TODO: this isn't thread safe. I may need to think more on concurrent vertex creation
            vertexUniqueSequenceCounter++;
            return vertexUniqueSequenceCounter;
        }

        // vertex aka node
        // todo: in a perfect world, instead of having multiple constructors
        //  where one is for full layers and one is for convolution layers
        //  maybe I should use inheritance to have more than one type of NNVertex.
        class NNVertex : public enable_shared_from_this<NNVertex> {
        public:
            // used for non-convolutional layers
            explicit NNVertex(const weak_ptr<HappymlDSL> &parent, LayerType layerType, const vector<size_t> &input_shape,
                              const vector<size_t> &output_shape, bool for_output, bool givenInput,
                              ActivationType activation_type, uint32_t vertexUniqueId) {
                this->parent = parent;
                this->node_type = layerType;
                this->activation_type = activation_type;
                this->inputShapes = {input_shape};
                this->outputShape = output_shape;
                this->bits = 32;
                this->use_bias = false;
                this->use_l2_regularization = node_type == LayerType::full;
                this->use_normalization = false;
                this->materialized = false;
                this->first_node = nullptr;
                this->producesOutput = for_output;
                this->kernel_size = 0;
                this->filters = 0;
                this->vertexUniqueId = vertexUniqueId;
                this->acceptsInput = givenInput;
                this->use_norm_clipping = false;
                this->norm_clipping_threshold = 5.0f;
                this->dropout_rate = 0.0f;
                if (for_output && node_type != LayerType::full) {
                    throw runtime_error("Only full or convolution2dValid layers can be used as output.");
                }
            }

            // used for convolutional layers
            explicit NNVertex(const weak_ptr<HappymlDSL> &parent, LayerType layerType, const vector<size_t> &input_shape,
                              const size_t filters, const size_t kernel_size, bool for_output, bool acceptsInput,
                              ActivationType activation_type, uint32_t vertexUniqueId) {
                this->parent = parent;
                this->node_type = layerType;
                this->activation_type = activation_type;
                this->inputShapes = {input_shape};
                this->outputShape = {input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, filters};
                this->bits = 32;
                this->use_bias = false;
                this->use_l2_regularization = false;
                this->use_normalization = false;
                this->materialized = true;
                this->first_node = nullptr;
                this->kernel_size = kernel_size;
                this->filters = filters;
                this->producesOutput = for_output;
                this->vertexUniqueId = vertexUniqueId;
                this->acceptsInput = acceptsInput;
                this->use_norm_clipping = false;
                this->norm_clipping_threshold = 5.0f;
                this->dropout_rate = 0.0f;
                if (for_output && node_type != LayerType::convolution2dValid) {
                    throw runtime_error("Only full or convolution2dValid layers can be used as output.");
                }
            }

            // used for concatenate layers
            explicit NNVertex(const weak_ptr<HappymlDSL> &parent, LayerType layerType, const vector<vector<size_t>> &input_shapes,
                              const vector<size_t> &output_shape, uint32_t vertexUniqueId) {
                this->parent = parent;
                this->node_type = layerType;
                this->activation_type = ActivationType::linear;
                this->inputShapes = input_shapes;
                this->outputShape = output_shape;
                this->bits = 32;
                this->use_bias = false;
                this->use_l2_regularization = false;
                this->use_normalization = false;
                this->materialized = false;
                this->first_node = nullptr;
                this->producesOutput = false;
                this->kernel_size = 0;
                this->filters = 0;
                this->vertexUniqueId = vertexUniqueId;
                this->acceptsInput = false;
                this->use_norm_clipping = false;
                this->norm_clipping_threshold = 5.0f;
                this->dropout_rate = 0.0f;
            }

            shared_ptr<NNVertex> setUseL2Regularization(bool b) {
                this->use_l2_regularization = b;
                if (b && (node_type != LayerType::full && node_type != LayerType::convolution2dValid)) {
                    throw runtime_error("L2 regularization can only be used on full layers or convolution2dValid layers");
                }
                return shared_from_this();
            }

            shared_ptr<NNVertex> setUseNormalization(bool b) {
                this->use_normalization = b;
                if (b && (node_type != LayerType::full && node_type != LayerType::convolution2dValid)) {
                    throw runtime_error("Layer Normalization can only be used on full layers or convolution2dValid layers");
                }
                return shared_from_this();
            }

            shared_ptr<NNVertex> setUseBias(bool b) {
                this->use_bias = b;
                if (b && (node_type != LayerType::full &&
                          node_type != LayerType::convolution2dValid)) {
                    throw runtime_error("Bias can only be used on full or convolution2dValid layers");
                }
                return shared_from_this();
            }

            shared_ptr<NNVertex> setBits(uint8_t bits_val) {
                this->bits = bits_val;
                return shared_from_this();
            }

            shared_ptr<NNVertex> setMaterialized(bool m) {
                this->materialized = m;
                if (m && (node_type != LayerType::full &&
                          node_type != LayerType::convolution2dValid)) {
                    throw runtime_error("Materialized can only be used on full or convolution2dValid layers");
                }
                return shared_from_this();
            }

            shared_ptr<NNVertex> setUseNormClipping(bool b) {
                this->use_norm_clipping = b;
                return shared_from_this();
            }

            shared_ptr<NNVertex> setNormClippingThreshold(float value) {
                this->norm_clipping_threshold = value;
                return setUseNormClipping(true);
            }

            // edge aka connection
            struct NNEdge {
                weak_ptr<NNVertex> from;
                shared_ptr<NNVertex> to;
            };

            shared_ptr<NNVertex> addOutput(const size_t nodeOutputShape, ActivationType activationType) {
                return addLayer(this->outputShape, {1, nodeOutputShape, 1}, LayerType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutputLayer(const vector<size_t> &nodeOutputShape, ActivationType activationType) {
                return addLayer(this->outputShape, nodeOutputShape, LayerType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &nodeOutputShape, const size_t outputKernelSize,
                                           LayerType layerType, ActivationType activationType) {
                // todo: support other types of convolution nodes here
                if (layerType != LayerType::convolution2dValid) {
                    throw runtime_error("Only convolutional nodes have a kernel size.");
                }
                auto result = addLayer(nodeOutputShape[2], outputKernelSize,
                                       layerType, true,
                                       activationType);
                if (result->outputShape[0] != nodeOutputShape[0] ||
                    result->outputShape[1] != nodeOutputShape[1] ||
                    result->outputShape[2] != nodeOutputShape[2]) {
                    stringstream ss;
                    ss << "The calculated output shape of the node ("
                       << result->outputShape[0] << ", " << result->outputShape[1] << ", " << result->outputShape[2]
                       << ") didn't match the desired output shape ("
                       << nodeOutputShape[0] << ", " << nodeOutputShape[1] << ", " << nodeOutputShape[2] << ")";

                    // todo: could we have taken other action here to avoid an error and reshape the output?
                    throw runtime_error(ss.str().c_str());
                }
                return result;
            }

            shared_ptr<NNVertex> addLayer(const size_t nodeOutputShape,
                                          LayerType layerType, ActivationType activationType) {
                return addLayer({1, nodeOutputShape, 1}, layerType, activationType);
            }

            shared_ptr<NNVertex> addLayer(const vector<size_t> &nodeOutputShape, LayerType layerType,
                                          ActivationType activationType) {
                return addLayer(this->outputShape, nodeOutputShape,
                                layerType, false, activationType);
            }

            shared_ptr<NNVertex> addLayer(const size_t next_filters, const size_t next_kernel_size, LayerType layerType,
                                          ActivationType activationType) {
                return addLayer(next_filters, next_kernel_size, layerType, false, activationType);
            }

            shared_ptr<NNVertex> addLayer(const size_t next_filters, const size_t next_kernel_size,
                                          LayerType layerType, bool next_for_output, ActivationType activationType) {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, layerType, this->outputShape,
                                                 next_filters, next_kernel_size,
                                                 next_for_output, false,
                                                 activationType, parentObject->nextVertexId());
                addEdge(nnv);
                return nnv;
            }

            shared_ptr<NNVertex> addDropoutLayer(float dropout_rate=0.05f) {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, LayerType::dropout, this->outputShape,
                                                 this->outputShape,
                                                 false, false,
                                                 ActivationType::linear, parentObject->nextVertexId());
                addEdge(nnv);
                return nnv;
            }

            shared_ptr<NNVertex> addNormalizationLayer() {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, LayerType::normalize, this->outputShape,
                                                 this->outputShape,
                                                 false, false,
                                                 ActivationType::linear, parentObject->nextVertexId());
                addEdge(nnv);
                return nnv;
            }

            shared_ptr<NNVertex> addLayer(const vector<size_t> &nodeInputShape,
                                          const vector<size_t> &nodeOutputShape, LayerType layerType,
                                          bool next_for_output, ActivationType activationType) {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, layerType, nodeInputShape, nodeOutputShape,
                                                 next_for_output, false,
                                                 activationType, parentObject->nextVertexId());
                addEdge(nnv);
                return nnv;
            }

            void addEdge(const shared_ptr<NNVertex> &to) {
                auto nne = make_shared<NNEdge>();
                nne->from = shared_from_this();
                nne->to = to;
                edges.push_back(nne);
            }

            void reset() {
                this->first_node = nullptr;
            }

            shared_ptr<NeuralNetworkForTraining> build() {
                return parent.lock()->build();
            }

            shared_ptr<NeuralNetworkNode> buildLayer(const shared_ptr<NeuralNetworkForTraining> &nn,
                                                     vector<vector<string>> &networkMetadata) {
                if (first_node) {
                    // this node has already been built, don't infinitely recurse.
                    return first_node;
                }
                vector<string> metadata_row;
                metadata_row.emplace_back("vertex");
                metadata_row.push_back(asString(getVertexUniqueId()));
                metadata_row.push_back(asString(doesAcceptInput()));
                metadata_row.push_back(asString(isForOutput()));
                metadata_row.push_back(nodeTypeToString(getNodeType()));
                metadata_row.push_back(activationTypeToString(getActivationType()));
                metadata_row.push_back(asString(isMaterialized()));
                metadata_row.push_back(asString(isUseBias()));
                metadata_row.push_back(asString(getBits()));
                metadata_row.push_back(asString(inputShapes.size()));
                for (auto &inputShape: inputShapes) {
                    metadata_row.push_back(asString(inputShape[0]));
                    metadata_row.push_back(asString(inputShape[1]));
                    metadata_row.push_back(asString(inputShape[2]));
                }
                metadata_row.push_back(asString(outputShape[0]));
                metadata_row.push_back(asString(outputShape[1]));
                metadata_row.push_back(asString(outputShape[2]));
                metadata_row.push_back(asString(getFilters()));
                metadata_row.push_back(asString(getKernelSize()));
                metadata_row.push_back(asString(isUseL2Regularization()));
                metadata_row.push_back(asString(isUseNormalization()));
                metadata_row.push_back(asString(isUseNormClipping()));
                metadata_row.push_back(asString(getNormClippingThreshold()));
                metadata_row.push_back(asString(getDropoutRate()));
                networkMetadata.push_back(metadata_row);
                shared_ptr<BaseOptimizer> optimizer = nn->getOptimizer();
                shared_ptr<NeuralNetworkNode> next_node;
                shared_ptr<NeuralNetworkNode> last_node = nullptr;
                if (node_type == LayerType::full) {
                    auto inputShape = inputShapes[0];
                    if (inputShape[0] > 1) {
                        auto flatten_node = make_shared<NeuralNetworkNode>(make_shared<FlattenLayer>());
                        last_node = appendNode(last_node, flatten_node);
                    }
                    string fullNodeLabel = asString(vertexUniqueId) + "_full";
                    const shared_ptr<FullyConnectedLayer> &fcn = make_shared<FullyConnectedLayer>(fullNodeLabel,
                                                                                                  inputShape[0] *
                                                                                                  inputShape[1] *
                                                                                                  inputShape[2],
                                                                                                  outputShape[0] *
                                                                                                  outputShape[1] *
                                                                                                  outputShape[2],
                                                                                                  bits,
                                                                                                  optimizer->registerForWeightChanges(),
                                                                                                  use_l2_regularization);
                    next_node = make_shared<NeuralNetworkNode>(fcn);
                } else if (node_type == LayerType::concatenate) {
                    // TODO: we could probably have a strategy for handling different input shapes.
                    //  Strategy 1: pad the smaller inputs to match the largest input. (Most memory efficient if there is convolution.)
                    //  Strategy 2: find the least common multiple (LCM) of the input shapes and repeat the inputs to match. (Best quality if there is convolution.)
                    //  Strategy 3: flatten all inputs to 1d vectors and concatenate them. (Best option if there is no convolution.)

                    // NOTE: At this point in the code, I think using LCM makes the most sense.
                    // If we had inputs to the model that were not using convolution layers, they would have been flattened by
                    // now. If we have inputs that are using convolution layers, we should use the LCM to make sure that the
                    // inputs will be able to be concatenated.
                    string concatNodeLabel = asString(vertexUniqueId) + "_concat";
                    auto concat_node = make_shared<ConcatenateWideLayer>(concatNodeLabel, inputShapes);
                    next_node = make_shared<NeuralNetworkNode>(concat_node);
                } else if (node_type == LayerType::flatten) {
                    next_node = make_shared<NeuralNetworkNode>(make_shared<FlattenLayer>());
                } else if (node_type == LayerType::normalize) {
                    next_node = make_shared<NeuralNetworkNode>(make_shared<NormalizationLayer>());
                } else if (node_type == LayerType::dropout) {
                    auto inputShape = inputShapes[0];
                    if (inputShape[0] > 1) {
                        auto flatten_node = make_shared<NeuralNetworkNode>(make_shared<FlattenLayer>());
                        last_node = appendNode(last_node, flatten_node);
                    }
                    string dropoutNodeLabel = asString(vertexUniqueId) + "_dropout";
                    next_node = make_shared<NeuralNetworkNode>(make_shared<DropoutLayer>(dropoutNodeLabel, inputShape, dropout_rate));
                } else if (node_type == LayerType::convolution2dValid) {
                    auto inputShape = inputShapes[0];
                    string c2dvLabel = asString(vertexUniqueId) + "_c2dv";
                    auto c2d = make_shared<Convolution2dValidFunction>(c2dvLabel, inputShape, filters, kernel_size,
                                                                       bits,
                                                                       optimizer->registerForWeightChanges(),
                                                                       use_l2_regularization);
                    next_node = make_shared<NeuralNetworkNode>(c2d);
                } else {
                    throw runtime_error("Unimplemented NodeType");
                }
                if (use_norm_clipping) {
                    next_node->set_use_norm_clipping(use_norm_clipping);
                    next_node->set_norm_clipping_threshold(norm_clipping_threshold);
                }
                last_node = appendNode(last_node, next_node);

                if (use_bias) {
                    string biasLabel = asString(vertexUniqueId) + "_bias";
                    auto b = make_shared<BiasLayer>(biasLabel, outputShape, outputShape, bits, optimizer->registerForBiasChanges());
                    auto bias_node = make_shared<NeuralNetworkNode>(b);
                    if (use_norm_clipping) {
                        bias_node->set_use_norm_clipping(use_norm_clipping);
                        bias_node->set_norm_clipping_threshold(norm_clipping_threshold);
                    }
                    last_node = appendNode(last_node, bias_node);
                }

                if (use_normalization) {
                    auto norm = make_shared<NormalizationLayer>();
                    auto norm_node = make_shared<NeuralNetworkNode>(norm);
                    if (use_norm_clipping) {
                        norm_node->set_use_norm_clipping(use_norm_clipping);
                        norm_node->set_norm_clipping_threshold(norm_clipping_threshold);
                    }
                    last_node = appendNode(last_node, norm_node);
                }

                if (node_type == LayerType::convolution2dValid || node_type == LayerType::full) {
                    shared_ptr<ActivationFunction> activationFunction = createActivationFunction();
                    auto activation_node = make_shared<NeuralNetworkOutputNode>(
                            make_shared<ActivationLayer>(activationFunction));
                    if (use_norm_clipping) {
                        activation_node->set_use_norm_clipping(use_norm_clipping);
                        activation_node->set_norm_clipping_threshold(norm_clipping_threshold);
                    }
                    last_node = appendNode(last_node, activation_node);
                    if (producesOutput) {
                        nn->addOutput(activation_node);
                    }
                    last_node->setMaterialized(materialized);
                }

                vector<string> edgeMetadata{"edge", asString(getVertexUniqueId())};
                for (const auto &edge: edges) {
                    edgeMetadata.push_back(asString(edge->to->getVertexUniqueId()));
                    auto childNode = edge->to->buildLayer(nn, networkMetadata);
                    last_node->add(childNode);
                }
                if (edgeMetadata.size() > 2) {
                    networkMetadata.push_back(edgeMetadata);
                }
                return first_node;
            }

            shared_ptr<ActivationFunction> createActivationFunction() const {
                shared_ptr<ActivationFunction> activationFunction;
                switch (activation_type) {
                    case tanhDefault:
                        activationFunction = make_shared<TanhActivationFunction>();
                        break;
                    case relu:
                        activationFunction = make_shared<ReLUActivationFunction>();
                        break;
                    case sigmoid:
                        activationFunction = make_shared<SigmoidActivationFunction>();
                        break;
                    case sigmoidApprox:
                        activationFunction = make_shared<SigmoidApproximationActivationFunction>();
                        break;
                    case tanhApprox:
                        activationFunction = make_shared<TanhApproximationActivationFunction>();
                        break;
                    case softmax:
                        activationFunction = make_shared<SoftmaxActivationFunction>();
                        break;
                    case linear:
                        activationFunction = make_shared<LinearActivationFunction>();
                        break;
                    case leaky:
                        activationFunction = make_shared<LeakyReLUActivationFunction>();
                        break;
                    default:
                        activationFunction = make_shared<ReLUActivationFunction>();
                }
                return activationFunction;
            }

            shared_ptr<NeuralNetworkNode> appendNode(const shared_ptr<NeuralNetworkNode> &last_node,
                                                     const shared_ptr<NeuralNetworkNode> &node) {
                if (!first_node) {
                    first_node = node;
                    return node;
                }
                return last_node->add(node);
            }

            bool doesAcceptInput() const {
                return acceptsInput;
            }

            uint32_t getVertexUniqueId() const {
                return vertexUniqueId;
            }

            bool isForOutput() const {
                return producesOutput;
            }

            LayerType getNodeType() {
                return node_type;
            }

            ActivationType getActivationType() {
                return activation_type;
            }

            bool isMaterialized() const {
                return materialized;
            }

            bool isUseBias() const {
                return use_bias;
            }

            bool isUseL2Regularization() const {
                return use_l2_regularization;
            }

            bool isUseNormalization() const {
                return use_normalization;
            }

            uint8_t getBits() const {
                return bits;
            }

            vector<vector<size_t>> getInputShapes() {
                return inputShapes;
            }

            vector<size_t> getOutputShape() {
                return outputShape;
            }

            size_t getFilters() const {
                return filters;
            }

            size_t getKernelSize() const {
                return kernel_size;
            }

            bool isUseNormClipping() const {
                return use_norm_clipping;
            }

            float getNormClippingThreshold() const {
                return norm_clipping_threshold;
            }

            shared_ptr<NNVertex> usingParent(const shared_ptr<HappymlDSL> &new_parent) {
                this->parent = new_parent;
                return shared_from_this();
            }

            float getDropoutRate() const {
                return dropout_rate;
            }

            void setDropoutRate(float dropout_rate) {
                this->dropout_rate = dropout_rate;
            }

        private:
            weak_ptr<HappymlDSL> parent;
            vector<shared_ptr<NNEdge>> edges;
            LayerType node_type;
            vector<vector<size_t>> inputShapes;
            vector<size_t> outputShape;
            ActivationType activation_type;
            bool materialized;
            bool use_bias;
            bool use_l2_regularization;
            bool use_normalization;
            bool use_norm_clipping;
            float norm_clipping_threshold;
            uint8_t bits;
            shared_ptr<NeuralNetworkNode> first_node;
            size_t kernel_size{};
            size_t filters{};
            bool producesOutput;
            bool acceptsInput;
            uint32_t vertexUniqueId;
            float dropout_rate;
        };

        shared_ptr<NNVertex> addInputLayer(const size_t input_shape, const size_t output_shape, LayerType layerType,
                                           ActivationType activationType) {
            return addInputLayer({1, input_shape, 1}, {1, output_shape, 1}, layerType, activationType);
        }

        shared_ptr<NNVertex> addInputLayer(const vector<size_t> &input_shape,
                                           const vector<size_t> &output_shape, LayerType layerType,
                                           ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), layerType, input_shape,
                                             output_shape, false, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutputLayer(const vector<size_t> &input_shape,
                                                 const vector<size_t> &output_shape, LayerType layerType,
                                                 ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), layerType, input_shape,
                                             output_shape, true, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        // where kernel_size is the width and height of the convolution window being applied to the input
        // filters is the same as the depth of the output
        shared_ptr<NNVertex> addInputLayer(const vector<size_t> &input_shape,
                                           const size_t filters, const size_t kernel_size,
                                           LayerType layerType,
                                           ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), layerType, input_shape,
                                             filters, kernel_size, false, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutputLayer(const vector<size_t> &input_shape,
                                                 const size_t filters, const size_t kernel_size,
                                                 LayerType layerType, ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), layerType, input_shape,
                                             filters, kernel_size, true, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputLayer(const size_t input_shape, const vector<size_t> &output_shape, LayerType layerType,
                                           ActivationType activationType) {
            return addInputLayer({1, input_shape, 1}, output_shape, layerType, activationType);
        }

        shared_ptr<NNVertex> addInputLayer(const vector<size_t> &input_shape, const size_t output_shape, LayerType layerType,
                                           ActivationType activationType) {
            return addInputLayer(input_shape, {1, output_shape, 1}, layerType, activationType);
        }


        // TODO: all input today immediately feed into a layer of specified type. This worked great for very linear
        //  models that went from input->layer->layer->output. However, with multiple inputs, this feels a little
        //  weird. "Wait, so I have 3 inputs and you jammed them together into a dense layer?"
        //  I'm returning the concatenated layer below rather than making a full layer, but now it is inconsistent
        //  with the other add_layer methods.

        // flattens the input, concatenates them, and sends them to
        // a full layer which produces a single output of the specified shape
        shared_ptr<NNVertex> add_concatenated_input_layer(const vector<vector<size_t>> &input_shapes) {
            if (input_shapes.size() < 2) {
                throw runtime_error("add_concatenated_input_layer requires multiple inputs");
            }

            vector<shared_ptr<NNVertex>> new_input_receptors;
            new_input_receptors.reserve(input_shapes.size());
            size_t total_input_width = 0;
            for (auto &input_shape: input_shapes) {
                size_t width = input_shape[0] * input_shape[1] * input_shape[2];
                total_input_width += width;
                shared_ptr<NNVertex> next_input = addInputLayer(input_shape, width, LayerType::flatten, ActivationType::linear);
                new_input_receptors.push_back(next_input);
            }
            return add_concatenated_layer(new_input_receptors);
        }

        shared_ptr<NNVertex> add_concatenated_layer(const vector<shared_ptr<NNVertex>> &previous_layers) {
            size_t total_input_width = 0;
            vector<vector<size_t>> input_shapes;
            for (auto &layer: previous_layers) {
                total_input_width += layer->getOutputShape()[1];
                input_shapes.push_back(layer->getOutputShape());
            }
            vector<size_t> concat_shape_vector = {1, total_input_width, 1};
            auto concatenator = make_shared<NNVertex>(shared_from_this(),
                                                      LayerType::concatenate, input_shapes,
                                                      concat_shape_vector, nextVertexId());
            // connect the input receptors to the concatenator:
            for (auto &inputReceptor: previous_layers) {
                inputReceptor->addEdge(concatenator);
            }
            // add the dense layer (we only support one layer type right now) that the user asked for:
            return concatenator;
        }

    private:
        OptimizerType optimizerType;
        LossType lossType;
        float learningRate;
        float biasLearningRate;
        vector<shared_ptr<NNVertex>> inputReceptors;
        string modelName;
        string repoRootPath;
        uint32_t vertexUniqueSequenceCounter;
    };

    shared_ptr<HappymlDSL> neuralNetworkBuilder(OptimizerType optimizerType, const string &modelName = "unnamed",
                                                const string &repoRootPath = "repo") {
        auto result = make_shared<HappymlDSL>(optimizerType, modelName, repoRootPath);
        return result;
    }

    shared_ptr<HappymlDSL> neuralNetworkBuilder() {
        return neuralNetworkBuilder(adam);
    }

    void createVertexFromMetadata(const shared_ptr<HappymlDSL> &dsl,
                                  const vector<string> &vertexMetadata,
                                  const shared_ptr<HappymlDSL::NNVertex> &parent,
                                  map<uint32_t, shared_ptr<HappymlDSL::NNVertex>> &createdVertexes,
                                  map<uint32_t, vector<string>> &vertexes,
                                  map<uint32_t, vector<uint32_t>> &edgeFromTo) {
        // "vertex", id, is input, is output, node type, activation type, materialized, uses bias, bits,
        // input rows, input columns, input channels, output rows, output columns, output channels, filters, kernels
        const uint32_t vertexId = stoul(vertexMetadata[1]);
        const auto acceptsInput = asBool(vertexMetadata[2]);
        const bool producesOutput = asBool(vertexMetadata[3]);
        const LayerType layerType = stringToNodeType(vertexMetadata[4]);
        const ActivationType activationType = stringToActivationType(vertexMetadata[5]);
        const bool isMaterialized = asBool(vertexMetadata[6]);
        const bool useBias = asBool(vertexMetadata[7]);
        const uint8_t bits = stoul(vertexMetadata[8]);
        const size_t number_of_inputs = stoull(vertexMetadata[9]);
        size_t current_metadata_offset = 10;
        vector<vector<size_t>> inputShapes;
        for (size_t i = 0; i < number_of_inputs; ++i) {
            const vector<size_t> inputShape = {stoull(vertexMetadata[current_metadata_offset]),
                                               stoull(vertexMetadata[current_metadata_offset + 1]),
                                               stoull(vertexMetadata[current_metadata_offset + 2])};
            inputShapes.push_back(inputShape);
            current_metadata_offset += 3;
        }
        const vector<size_t> outputShape = {stoull(vertexMetadata[current_metadata_offset]),
                                            stoull(vertexMetadata[current_metadata_offset + 1]),
                                            stoull(vertexMetadata[current_metadata_offset + 2])};
        current_metadata_offset += 3;
        size_t filters = stoull(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        size_t kernels = stoull(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        bool use_l2_regularization = asBool(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        bool use_normalization = asBool(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        bool use_clipping = asBool(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        float clipping_threshold = stof(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;
        float dropout_rate = stof(vertexMetadata[current_metadata_offset]);
        current_metadata_offset++;

        if (acceptsInput) {
            if (inputShapes.size() > 1) {
                throw runtime_error("An input layer can't have multiple inputs.");
            }
            if (producesOutput) {
                if (filters > 0) {
                    createdVertexes[vertexId] = dsl->addInputOutputLayer(inputShapes[0], filters, kernels, layerType,
                                                                         activationType);
                } else {
                    createdVertexes[vertexId] = dsl->addInputOutputLayer(inputShapes[0], outputShape, layerType,
                                                                         activationType);
                }
            } else {
                if (filters > 0) {
                    createdVertexes[vertexId] = dsl->addInputLayer(inputShapes[0], filters, kernels, layerType,
                                                                   activationType);
                } else {
                    createdVertexes[vertexId] = dsl->addInputLayer(inputShapes[0], outputShape, layerType,
                                                                   activationType);
                }
            }
        } else {
            if (!parent) {
                throw runtime_error("missing parent");
            }

            if (layerType == concatenate) {
                // a concatenated layer has multiple parents.

                // because layers are in the order they were originally created and the concatenation
                // layer is last, if we are at this point, all parents should exist, we just need to find them
                // I think the parent order should also be maintained
                bool all_parents_exist = true;
                vector<shared_ptr<HappymlDSL::NNVertex>> all_parents;
                for (const auto &edge_pair: edgeFromTo) {
                    auto edge_to_ids = edge_pair.second;
                    // if edge_to_ids contains vertexId, then we need to add the parent to the concatenated layer
                    if (std::find(edge_to_ids.begin(), edge_to_ids.end(), vertexId) != edge_to_ids.end()) {
                        auto edge_from_id = edge_pair.first;
                        if (createdVertexes.count(edge_from_id) == 0) {
                            all_parents_exist = false;
                            break;
                        }
                        auto edge_from_vertex = createdVertexes[edge_from_id];
                        all_parents.push_back(edge_from_vertex);
                    }
                }
                if (!all_parents_exist) {
                    return;
                }
                createdVertexes[vertexId] = dsl->add_concatenated_layer(all_parents);
            } else if (filters > 0) {
                createdVertexes[vertexId] = parent->addLayer(filters,
                                                             kernels,
                                                             layerType,
                                                             producesOutput,
                                                             activationType);
            } else {
                if (producesOutput && layerType != full) {
                    throw runtime_error("output node type wasn't full");
                }
                createdVertexes[vertexId] = parent->addLayer(inputShapes[0],
                                                             outputShape,
                                                             layerType,
                                                             producesOutput,
                                                             activationType);
            }
        }
        createdVertexes[vertexId]->setMaterialized(isMaterialized);
        createdVertexes[vertexId]->setUseBias(useBias);
        createdVertexes[vertexId]->setBits(bits);
        createdVertexes[vertexId]->setUseL2Regularization(use_l2_regularization);
        createdVertexes[vertexId]->setUseNormalization(use_normalization);
        createdVertexes[vertexId]->setNormClippingThreshold(clipping_threshold); // set this first because it would enable clipping
        createdVertexes[vertexId]->setUseNormClipping(use_clipping); // set this second because it could disable clipping if required
        createdVertexes[vertexId]->setDropoutRate(dropout_rate);

        if (edgeFromTo.count(vertexId) > 0) {
            auto edges = edgeFromTo[vertexId];
            for (auto nextEdge: edges) {
                if (vertexes.count(nextEdge) < 1) {
                    throw runtime_error("Bad model definition. Edge not found!");
                }
                auto nextVertexMetadata = vertexes[nextEdge];
                createVertexFromMetadata(dsl,
                                         nextVertexMetadata,
                                         createdVertexes[vertexId],
                                         createdVertexes,
                                         vertexes,
                                         edgeFromTo);
            }
        }
    }

    shared_ptr<NeuralNetworkForTraining> loadNeuralNetworkForTraining(const string &modelName,
                                                                      const string &repoRootPath = "repo") {
        string modelPath = repoRootPath + "/" + modelName;
        string configPath = modelPath + "/model.config";
        // if model.config doesn't exist the model failed during training
        auto configReader = make_shared<DelimitedTextFileReader>(configPath, ':');

        auto optimizerRecord = configReader->nextRecord();
        if (optimizerRecord[0] != "optimizer") {
            throw runtime_error("Invalid model.config missing optimizer field.");
        }
        const OptimizerType optimizerType = stringToOptimizerType(optimizerRecord[1]);

        auto learningRateRecord = configReader->nextRecord();
        if (learningRateRecord[0] != "learningRate") {
            throw runtime_error("Invalid model.config missing learningRate field.");
        }
        float learningRate = stof(learningRateRecord[1]);

        auto biasLearningRateRecord = configReader->nextRecord();
        if (biasLearningRateRecord[0] != "biasLearningRate") {
            throw runtime_error("Invalid model.config missing biasLearningRate field.");
        }
        float biasLearningRate = stof(biasLearningRateRecord[1]);

        auto lossRecord = configReader->nextRecord();
        if (lossRecord[0] != "loss") {
            throw runtime_error("Invalid model.config missing loss field.");
        }
        const LossType lossType = stringToLossType(lossRecord[1]);

        auto dsl = neuralNetworkBuilder(optimizerType, modelName, repoRootPath)
                ->setLossFunction(lossType)
                ->setLearningRate(learningRate)
                ->setBiasLearningRate(biasLearningRate);

        set<uint32_t> headVertexes;
        map<uint32_t, vector<string>> vertexes;
        map<uint32_t, vector<uint32_t>> edgeFromTo;
        while (configReader->hasNext()) {
            auto nextRecord = configReader->nextRecord();
            if (nextRecord[0] == "vertex") {
                uint32_t vertexId = stoul(nextRecord[1]);
                vertexes[vertexId] = nextRecord;
                const auto acceptsInput = asBool(nextRecord[2]);
                if (acceptsInput) {
                    headVertexes.insert(vertexId);
                }
            } else {
                uint32_t fromLabel = stoul(nextRecord[1]);
                vector<uint32_t> toLabels;
                for (size_t index = 2; index < nextRecord.size(); index++) {
                    uint32_t toLabel = stoul(nextRecord[index]);
                    toLabels.push_back(toLabel);
                }
                edgeFromTo[fromLabel] = toLabels;
            }
        }
        map<uint32_t, shared_ptr<HappymlDSL::NNVertex>> createdVertexes;
        for (auto headVertexId: headVertexes) {
            auto vertexMetadata = vertexes[headVertexId];
            createVertexFromMetadata(dsl,
                                     vertexMetadata,
                                     nullptr,
                                     createdVertexes,
                                     vertexes,
                                     edgeFromTo);
        }

        auto resultNeuralNetwork = dsl->build();
        resultNeuralNetwork->loadKnowledge("default");
        return resultNeuralNetwork;
    }

}

#endif //HAPPYML_HAPPYML_DSL_HPP
