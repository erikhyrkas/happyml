//
// Created by Erik Hyrkas on 11/2/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MODEL_HPP
#define HAPPYML_MODEL_HPP

#include <iostream>
#include <chrono>
#include "enums.hpp"
#include "optimizer_factory.hpp"
#include "neural_network.hpp"
#include "../training_data/training_dataset.hpp"

using namespace happyml;
using namespace std;

namespace happymldsl {

    class HappymlDSL : public enable_shared_from_this<HappymlDSL> {
    public:
        explicit HappymlDSL(OptimizerType optimizerType, const string &modelName = "unnamed",
                            const string &repoRootPath = "repo") {
            this->optimizerType = optimizerType;
            this->modelName = modelName;
            if (!std::all_of(modelName.begin(), modelName.end(),
                             [](int c) { return std::isalnum(c) || c == '_'; })) {
                throw exception("Model name must contain only alphanumeric characters.");
            }
            switch (optimizerType) {
                case microbatch:
                    this->learningRate = 0.1;
                    this->biasLearningRate = 0.01;
                    break;
                case adam:
                    this->learningRate = 0.01;
                    this->biasLearningRate = 0.001;
                    break;
                case sgdm:
                    this->learningRate = 0.01;
                    this->biasLearningRate = 0.001;
                    break;
                default:
                    throw exception("Unsupported model type.");
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
                throw exception("Model name must contain only alphanumeric characters.");
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
                neuralNetwork->addHeadNode(head->buildNode(neuralNetwork, networkMetadata));
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
            NNVertex(const weak_ptr<HappymlDSL> &parent, NodeType nodeType, const vector<size_t> &input_shape,
                     const vector<size_t> &output_shape, bool for_output, bool givenInput,
                     ActivationType activation_type, uint32_t vertexUniqueId) {
                this->parent = parent;
                this->node_type = nodeType;
                this->activation_type = activation_type;
                this->inputShape = input_shape;
                this->outputShape = output_shape;
                this->bits = 32;
                this->use_bias = true;
                this->materialized = false;
                this->first_node = nullptr;
                this->producesOutput = for_output;
                this->kernel_size = 0;
                this->filters = 0;
                this->vertexUniqueId = vertexUniqueId;
                this->acceptsInput = givenInput;
            }

            // used for convolutional layers
            NNVertex(const weak_ptr<HappymlDSL> &parent, NodeType nodeType, const vector<size_t> &input_shape,
                     const size_t filters, const size_t kernel_size, bool for_output, bool acceptsInput,
                     ActivationType activation_type, uint32_t vertexUniqueId) {
                this->parent = parent;
                this->node_type = nodeType;
                this->activation_type = activation_type;
                this->inputShape = input_shape;
                this->outputShape = {input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, filters};
                this->bits = 32;
                this->use_bias = true;
                this->materialized = true;
                this->first_node = nullptr;
                this->kernel_size = kernel_size;
                this->filters = filters;
                this->producesOutput = for_output;
                this->vertexUniqueId = vertexUniqueId;
                this->acceptsInput = acceptsInput;
            }

            shared_ptr<NNVertex> setUseBias(bool b) {
                this->use_bias = b;
                return shared_from_this();
            }

            shared_ptr<NNVertex> setBits(uint8_t bits_val) {
                this->bits = bits_val;
                return shared_from_this();
            }

            shared_ptr<NNVertex> setMaterialized(bool m) {
                this->materialized = m;
                return shared_from_this();
            }

            // edge aka connection
            struct NNEdge {
                weak_ptr<NNVertex> from;
                shared_ptr<NNVertex> to;
            };

            shared_ptr<NNVertex> addOutput(const size_t nodeOutputShape, ActivationType activationType) {
                return addNode(this->outputShape, {1, nodeOutputShape, 1}, NodeType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &nodeOutputShape, ActivationType activationType) {
                return addNode(this->outputShape, nodeOutputShape, NodeType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &nodeOutputShape, const size_t outputKernelSize,
                                           NodeType nodeType, ActivationType activationType) {
                // todo: support other types of convolution nodes here
                if (nodeType != NodeType::convolution2dValid) {
                    throw exception("Only convolutional nodes have a kernel size.");
                }
                auto result = addNode(nodeOutputShape[2], outputKernelSize,
                                      nodeType, true,
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
                    throw exception(ss.str().c_str());
                }
                return result;
            }

            shared_ptr<NNVertex> addNode(const size_t nodeOutputShape,
                                         NodeType nodeType, ActivationType activationType) {
                return addNode({1, nodeOutputShape, 1}, nodeType, activationType);
            }

            shared_ptr<NNVertex> addNode(const vector<size_t> &nodeOutputShape, NodeType nodeType,
                                         ActivationType activationType) {
                return addNode(this->outputShape, nodeOutputShape,
                               nodeType, false, activationType);
            }

            shared_ptr<NNVertex> addNode(const size_t next_filters, const size_t next_kernel_size, NodeType nodeType,
                                         ActivationType activationType) {
                return addNode(next_filters, next_kernel_size, nodeType, false, activationType);
            }

            shared_ptr<NNVertex> addNode(const size_t next_filters, const size_t next_kernel_size,
                                         NodeType nodeType, bool next_for_output, ActivationType activationType) {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, nodeType, this->outputShape,
                                                 next_filters, next_kernel_size,
                                                 next_for_output, false,
                                                 activationType, parentObject->nextVertexId());
                auto nne = make_shared<NNEdge>();
                nne->from = shared_from_this();
                nne->to = nnv;
                edges.push_back(nne);
                return nnv;
            }

            shared_ptr<NNVertex> addNode(const vector<size_t> &nodeInputShape,
                                         const vector<size_t> &nodeOutputShape, NodeType nodeType,
                                         bool next_for_output, ActivationType activationType) {
                auto parentObject = parent.lock();
                auto nnv = make_shared<NNVertex>(parentObject, nodeType, nodeInputShape, nodeOutputShape,
                                                 next_for_output, false,
                                                 activationType, parentObject->nextVertexId());
                auto nne = make_shared<NNEdge>();
                nne->from = shared_from_this();
                nne->to = nnv;
                edges.push_back(nne);
                return nnv;
            }

            void reset() {
                this->first_node = nullptr;
            }

            shared_ptr<NeuralNetworkForTraining> build() {
                return parent.lock()->build();
            }

            shared_ptr<NeuralNetworkNode> buildNode(const shared_ptr<NeuralNetworkForTraining> &nn,
                                                    vector<vector<string>> &networkMetadata) {
                if (first_node) {
                    // this node has already been built, don't infinitely recurse.
                    return first_node;
                }
                networkMetadata.push_back({"vertex",
                                           asString(getVertexUniqueId()),
                                           asString(doesAcceptInput()),
                                           asString(isForOutput()),
                                           nodeTypeToString(getNodeType()),
                                           activationTypeToString(getActivationType()),
                                           asString(isMaterialized()),
                                           asString(isUseBias()),
                                           asString(getBits()),
                                           asString(inputShape[0]),
                                           asString(inputShape[1]),
                                           asString(inputShape[2]),
                                           asString(outputShape[0]),
                                           asString(outputShape[1]),
                                           asString(outputShape[2]),
                                           asString(getFilters()),
                                           asString(getKernelSize())
                                          });
                shared_ptr<BaseOptimizer> optimizer = nn->getOptimizer();
                shared_ptr<NeuralNetworkNode> next_node;
                shared_ptr<NeuralNetworkNode> last_node = nullptr;
                if (node_type == NodeType::full) {
                    if (inputShape[0] > 1) {
                        auto flatten_node = make_shared<NeuralNetworkNode>(make_shared<NeuralNetworkFlattenFunction>());
                        last_node = appendNode(last_node, flatten_node);
                    }
                    string fullNodeLabel = asString(vertexUniqueId) + "_full";
                    const shared_ptr<FullyConnectedNeurons> &fcn = make_shared<FullyConnectedNeurons>(fullNodeLabel,
                                                                                                      inputShape[0] *
                                                                                                      inputShape[1] *
                                                                                                      inputShape[2],
                                                                                                      outputShape[0] *
                                                                                                      outputShape[1] *
                                                                                                      outputShape[2],
                                                                                                      bits, optimizer);
                    next_node = make_shared<NeuralNetworkNode>(fcn);
                } else if (node_type == NodeType::convolution2dValid) {
                    string c2dvLabel = asString(vertexUniqueId) + "_c2dv";
                    auto c2d = make_shared<Convolution2dValidFunction>(c2dvLabel, inputShape, filters, kernel_size,
                                                                       bits,
                                                                       optimizer);
                    next_node = make_shared<NeuralNetworkNode>(c2d);
                } else {
                    throw exception("Unimplemented NodeType");
                }

                last_node = appendNode(last_node, next_node);

                if (use_bias) {
                    string biasLabel = asString(vertexUniqueId) + "_bias";
                    auto b = make_shared<BiasNeuron>(biasLabel, outputShape, outputShape, bits, optimizer);
                    auto bias_node = make_shared<NeuralNetworkNode>(b);
                    last_node = appendNode(last_node, bias_node);
                }

                shared_ptr<ActivationFunction> activationFunction = createActivationFunction();
                auto activation_node = make_shared<NeuralNetworkOutputNode>(
                        make_shared<NeuralNetworkActivationFunction>(activationFunction));
                last_node = appendNode(last_node, activation_node);

                if (producesOutput) {
                    nn->addOutput(activation_node);
                }

                last_node->setMaterialized(materialized);
                vector<string> edgeMetadata{"edge", asString(getVertexUniqueId())};
                for (const auto &edge: edges) {
                    edgeMetadata.push_back(asString(edge->to->getVertexUniqueId()));
                    auto childNode = edge->to->buildNode(nn, networkMetadata);
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

            NodeType getNodeType() {
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

            uint8_t getBits() const {
                return bits;
            }

            vector<size_t> getInputShape() {
                return inputShape;
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

        private:
            weak_ptr<HappymlDSL> parent;
            vector<shared_ptr<NNEdge>> edges;
            NodeType node_type;
            vector<size_t> inputShape;
            vector<size_t> outputShape;
            ActivationType activation_type;
            bool materialized;
            bool use_bias;
            uint8_t bits;
            shared_ptr<NeuralNetworkNode> first_node;
            size_t kernel_size{};
            size_t filters{};
            bool producesOutput;
            bool acceptsInput;
            uint32_t vertexUniqueId;
        };

        shared_ptr<NNVertex> addInput(const size_t input_shape, const size_t output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            return addInput({1, input_shape, 1}, {1, output_shape, 1}, nodeType, activationType);
        }

        shared_ptr<NNVertex> addInput(const vector<size_t> &input_shape,
                                      const vector<size_t> &output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape,
                                             output_shape, false, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutput(const vector<size_t> &input_shape,
                                            const vector<size_t> &output_shape, NodeType nodeType,
                                            ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape,
                                             output_shape, true, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        // where kernel_size is the width and height of the convolution window being applied to the input
        // filters is the same as the depth of the output
        shared_ptr<NNVertex> addInput(const vector<size_t> &input_shape,
                                      const size_t filters, const size_t kernel_size,
                                      NodeType nodeType,
                                      ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape,
                                             filters, kernel_size, false, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutput(const vector<size_t> &input_shape,
                                            const size_t filters, const size_t kernel_size,
                                            NodeType nodeType, ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape,
                                             filters, kernel_size, true, true,
                                             activationType, nextVertexId());
            inputReceptors.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInput(const size_t input_shape, const vector<size_t> &output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            return addInput({1, input_shape, 1}, output_shape, nodeType, activationType);
        }

        shared_ptr<NNVertex> addInput(const vector<size_t> &input_shape, const size_t output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            return addInput(input_shape, {1, output_shape, 1}, nodeType, activationType);
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
        return neuralNetworkBuilder();
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
        if (createdVertexes.count(vertexId) > 0) {
            // todo: need to add node combine functionality, so it is possible to concatenate,
            //  add, multiply, etc. inputs. When we make that functionality, this code will then need to
            //  be updated to support it. At the point I typed this, model dsl only supports a very linear
            //  approach of one node following another node.

            // might need to be something like:
            // parent->addMergeNode(NodeType::add, otherParentNode)
            return;
        }
        const auto acceptsInput = asBool(vertexMetadata[2]);
        const bool producesOutput = asBool(vertexMetadata[3]);
        const NodeType nodeType = stringToNodeType(vertexMetadata[4]);
        const ActivationType activationType = stringToActivationType(vertexMetadata[5]);
        const bool isMaterialized = asBool(vertexMetadata[6]);
        const bool useBias = asBool(vertexMetadata[7]);
        const uint8_t bits = stoul(vertexMetadata[8]);
        const vector<size_t> inputShape = {stoull(vertexMetadata[9]),
                                           stoull(vertexMetadata[10]),
                                           stoull(vertexMetadata[11])};
        const vector<size_t> outputShape = {stoull(vertexMetadata[12]),
                                            stoull(vertexMetadata[13]),
                                            stoull(vertexMetadata[14])};
        size_t filters = stoull(vertexMetadata[15]);
        size_t kernels = stoull(vertexMetadata[16]);
        if (acceptsInput) {
            if (producesOutput) {
                if (filters > 0) {
                    createdVertexes[vertexId] = dsl->addInputOutput(inputShape, filters, kernels, nodeType,
                                                                    activationType);
                } else {
                    createdVertexes[vertexId] = dsl->addInputOutput(inputShape, outputShape, nodeType,
                                                                    activationType);
                }
            } else {
                if (filters > 0) {
                    createdVertexes[vertexId] = dsl->addInput(inputShape, filters, kernels, nodeType,
                                                              activationType);
                } else {
                    createdVertexes[vertexId] = dsl->addInput(inputShape, outputShape, nodeType,
                                                              activationType);
                }
            }
        } else {
            if (!parent) {
                throw exception("missing parent");
            }

            if (filters > 0) {
                createdVertexes[vertexId] = parent->addNode(filters,
                                                            kernels,
                                                            nodeType,
                                                            producesOutput,
                                                            activationType);
            } else {
                if (nodeType != full) {
                    throw exception("output node type wasn't full");
                }
                createdVertexes[vertexId] = parent->addNode(inputShape,
                                                            outputShape,
                                                            nodeType,
                                                            producesOutput,
                                                            activationType);
            }
        }
        createdVertexes[vertexId]->setMaterialized(isMaterialized);
        createdVertexes[vertexId]->setUseBias(useBias);
        createdVertexes[vertexId]->setBits(bits);

        if (edgeFromTo.count(vertexId) > 0) {
            auto edges = edgeFromTo[vertexId];
            for (auto nextEdge: edges) {
                if (vertexes.count(nextEdge) < 1) {
                    throw exception("Bad model definition. Edge not found!");
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
        string configPath = modelPath + "/configuration.happyml";
        auto configReader = make_shared<DelimitedTextFileReader>(configPath, ':');

        auto optimizerRecord = configReader->nextRecord();
        if (optimizerRecord[0] != "optimizer") {
            throw exception("Invalid configuration.happyml missing optimizer field.");
        }
        const OptimizerType optimizerType = stringToOptimizerType(optimizerRecord[1]);

        auto learningRateRecord = configReader->nextRecord();
        if (learningRateRecord[0] != "learningRate") {
            throw exception("Invalid configuration.happyml missing learningRate field.");
        }
        float learningRate = stof(learningRateRecord[1]);

        auto biasLearningRateRecord = configReader->nextRecord();
        if (biasLearningRateRecord[0] != "biasLearningRate") {
            throw exception("Invalid configuration.happyml missing biasLearningRate field.");
        }
        float biasLearningRate = stof(biasLearningRateRecord[1]);

        auto lossRecord = configReader->nextRecord();
        if (lossRecord[0] != "loss") {
            throw exception("Invalid configuration.happyml missing loss field.");
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

#endif //HAPPYML_MODEL_HPP
