//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_MODEL_HPP
#define MICROML_MODEL_HPP

#include <iostream>
#include "dataset.hpp"
#include "optimizer.hpp"
#include "mbgd_optimizer.hpp"

using namespace microml;

namespace micromldsl {

    enum ModelType {
        microbatch
    };
    enum LossType {
        mse
    };
    enum NodeType {
        full, output, convolution2d
    };
    enum ActivationType {
        relu, tanh, sigmoid, leaky, softmax, sigmoid_approx, tanh_approx
    };

    class MicromlDSL : public enable_shared_from_this<MicromlDSL> {
    public:
        explicit MicromlDSL(ModelType modelType) {
            this->modelType = modelType;
            switch (modelType) {
                case microbatch:
                    this->learning_rate = 0.1;
                    break;
                default:
                    this->learning_rate = 0.1;
            }
            this->loss_type = LossType::mse;
        }

        shared_ptr<MicromlDSL> setLearningRate(float learningRate) {
            this->learning_rate = learningRate;
            return shared_from_this();
        }

        shared_ptr<MicromlDSL> setLossFunction(LossType lossType) {
            this->loss_type = lossType;
            return shared_from_this();
        }

        shared_ptr<NeuralNetworkForTraining> build() {
            shared_ptr<LossFunction> lossFunction;
            switch (loss_type) {
                case LossType::mse:
                    lossFunction = make_shared<MeanSquaredErrorLossFunction>();
                    break;
                default:
                    lossFunction = make_shared<MeanSquaredErrorLossFunction>();
            }
            shared_ptr<Optimizer> optimizer;
            switch (modelType) {
                case ModelType::microbatch:
                    optimizer = make_shared<SGDOptimizer>(learning_rate);
                    break;
                default:
                    optimizer = make_shared<SGDOptimizer>(learning_rate);
            }

            auto neuralNetwork = make_shared<NeuralNetworkForTraining>(lossFunction, optimizer);
            for (const auto &head: heads) {
                neuralNetwork->addHead(head->build_node(neuralNetwork));
            }

            return neuralNetwork;
        }

        // vertex aka node
        class NNVertex : public enable_shared_from_this<NNVertex> {
        public:
            NNVertex(const weak_ptr<MicromlDSL> &parent, NodeType nodeType, const vector<size_t> &input_shape,
                     const vector<size_t> &output_shape,
                     ActivationType activation_type) {
                this->parent = parent;
                this->node_type = nodeType;
                this->activation_type = activation_type;
                this->input_shape = input_shape;
                this->output_shape = output_shape;
                this->bits = 32;
                this->use_bias = true;
                this->created = false;
                this->materialized = nodeType == NodeType::output || nodeType == NodeType::convolution2d;
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

            shared_ptr<NNVertex> addOutput(const size_t outputShape, ActivationType activationType) {
                return addNode(outputShape, NodeType::output, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &outputShape, ActivationType activationType) {
                return addNode(outputShape, NodeType::output, activationType);
            }

            shared_ptr<NNVertex> addNode(const size_t outputShape, NodeType nodeType, ActivationType activationType) {
                return addNode({1, outputShape, 1}, nodeType, activationType);
            }

            shared_ptr<NNVertex>
            addNode(const vector<size_t> &outputShape, NodeType nodeType, ActivationType activationType) {
                return addNode(this->output_shape, outputShape, nodeType, activationType);
            }

            shared_ptr<NNVertex>
            addNode(const vector<size_t> &inputShape, const vector<size_t> &outputShape, NodeType nodeType,
                    ActivationType activationType) {
                auto nnv = make_shared<NNVertex>(parent.lock(), nodeType, inputShape, outputShape, activationType);
                auto nne = make_shared<NNEdge>();
                nne->from = shared_from_this();
                nne->to = nnv;
                edges.push_back(nne);
                return nnv;
            }

            void reset() {
                created = false;
            }

            shared_ptr<NeuralNetworkForTraining> build() {
                return parent.lock()->build();
            }

            shared_ptr<NeuralNetworkNode> build_node(const shared_ptr<NeuralNetworkForTraining> &nn) {
                if (created) {
                    return first_node;
                }
                shared_ptr<Optimizer> optimizer = nn->getOptimizer();
                if (node_type == NodeType::full || node_type == NodeType::output) {
                    // todo: create flatten node if input shape isn't flat
                    first_node = nullptr;
                    if(input_shape[0] > 1) {
                        first_node = make_shared<NeuralNetworkNode>(make_shared<NeuralNetworkFlattenFunction>());
                    }
                    auto next_node = make_shared<NeuralNetworkNode>(
                            optimizer->createFullyConnectedNeurons(input_shape[0]*input_shape[1]*input_shape[2], output_shape[0]*output_shape[1]*output_shape[2], bits));
                    if(first_node) {
                       first_node->add(next_node);
                    } else {
                        first_node = next_node;
                    }
                    last_node = next_node;
                    if (use_bias) {
                        auto bias_node = make_shared<NeuralNetworkNode>(
                                optimizer->createBias(output_shape, output_shape, bits));
                        next_node->add(bias_node);
                        last_node = bias_node;
                    }
                }
                shared_ptr<ActivationFunction> activationFunction;
                switch (activation_type) {
                    case ActivationType::tanh:
                        activationFunction = make_shared<TanhActivationFunction>();
                        break;
                    case ActivationType::relu:
                        activationFunction = make_shared<ReLUActivationFunction>();
                        break;
                    case ActivationType::sigmoid:
                        activationFunction = make_shared<SigmoidActivationFunction>();
                        break;
                    case ActivationType::sigmoid_approx:
                        activationFunction = make_shared<SigmoidApproximationActivationFunction>();
                        break;
                    case ActivationType::tanh_approx:
                        activationFunction = make_shared<TanhApproximationActivationFunction>();
                        break;
                    case ActivationType::softmax:
                        activationFunction = make_shared<SoftmaxActivationFunction>();
                        break;
                    case ActivationType::leaky:
                        activationFunction = make_shared<LeakyReLUActivationFunction>();
                        break;
                    default:
                        activationFunction = make_shared<ReLUActivationFunction>();
                }
                if (node_type == NodeType::output) {
                    auto activation_node = make_shared<NeuralNetworkOutputNode>(
                            make_shared<NeuralNetworkActivationFunction>(activationFunction));
                    last_node->add(activation_node);
                    last_node = activation_node;
                    nn->addOutput(activation_node);
                } else {
                    auto activation_node = make_shared<NeuralNetworkNode>(
                            make_shared<NeuralNetworkActivationFunction>(activationFunction));
                    last_node->add(activation_node);
                    last_node = activation_node;
                }
                last_node->setMaterialized(materialized);
                created = true;
                for (const auto &edge: edges) {
                    auto child_first = edge->to->build_node(nn);
                    last_node->add(child_first);
                }
                return first_node;
            }

        private:
            weak_ptr<MicromlDSL> parent;
            vector<shared_ptr<NNEdge>> edges;
            NodeType node_type;
            vector<size_t> input_shape;
            vector<size_t> output_shape;
            ActivationType activation_type;
            bool materialized;
            bool use_bias;
            uint8_t bits;
            bool created;
            shared_ptr<NeuralNetworkNode> first_node;
            shared_ptr<NeuralNetworkNode> last_node;
        };

        shared_ptr<NNVertex> addInput(const size_t input_shape, const size_t output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            return addInput({1, input_shape, 1}, {1, output_shape, 1}, nodeType, activationType);
        }

        shared_ptr<NNVertex>
        addInput(const vector<size_t> &input_shape, const vector<size_t> &output_shape, NodeType nodeType,
                 ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape, output_shape, activationType);
            heads.push_back(nnv);
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
        ModelType modelType;
        LossType loss_type;
        float learning_rate;
        vector<shared_ptr<NNVertex>> heads;
    };

    shared_ptr<MicromlDSL> neuralNetworkBuilder(ModelType modelType) {
        auto result = make_shared<MicromlDSL>(modelType);
        return result;
    }

    shared_ptr<MicromlDSL> neuralNetworkBuilder() {
        return neuralNetworkBuilder(microbatch); // TODO: change to adam. just testing microbatch right now.
    }
}

#endif //MICROML_MODEL_HPP
