//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_MODEL_HPP
#define MICROML_MODEL_HPP

#include <iostream>
#include "../training_data/training_dataset.hpp"
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
        full, convolution2dValid
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
                    this->bias_learning_rate = 0.01;
                    break;
                default:
                    this->learning_rate = 0.1;
                    this->bias_learning_rate = 0.01;
            }
            this->loss_type = LossType::mse;
        }

        shared_ptr<MicromlDSL> setBiasLearningRate(float biasLearningRate) {
            this->bias_learning_rate = biasLearningRate;
            return shared_from_this();
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
                    optimizer = make_shared<SGDOptimizer>(learning_rate, bias_learning_rate);
                    break;
                default:
                    optimizer = make_shared<SGDOptimizer>(learning_rate, bias_learning_rate);
            }

            auto neuralNetwork = make_shared<NeuralNetworkForTraining>(lossFunction, optimizer);
            for (const auto &head: heads) {
                neuralNetwork->addHead(head->build_node(neuralNetwork));
            }

            return neuralNetwork;
        }

        // vertex aka node
        // todo: in a perfect world, instead of having multiple constructors
        //  where one is for full layers and one is for convolution layers
        //  maybe I should use inheritance to have more than one type of NNVertex.
        class NNVertex : public enable_shared_from_this<NNVertex> {
        public:
            // used for non-convolutional layers
            NNVertex(const weak_ptr<MicromlDSL> &parent, NodeType nodeType, const vector<size_t> &input_shape,
                     const vector<size_t> &output_shape, bool for_output,
                     ActivationType activation_type) {
                this->parent = parent;
                this->node_type = nodeType;
                this->activation_type = activation_type;
                this->input_shape = input_shape;
                this->outputShape = output_shape;
                this->bits = 32;
                this->use_bias = true;
                this->materialized = false;
                this->first_node = nullptr;
                this->for_output = for_output;
                this->kernel_size = 0;
                this->filters = 0;
            }

            // used for convolutional layers
            NNVertex(const weak_ptr<MicromlDSL> &parent, NodeType nodeType, const vector<size_t> &input_shape,
                     const size_t filters, const size_t kernel_size, bool for_output, ActivationType activation_type) {
                this->parent = parent;
                this->node_type = nodeType;
                this->activation_type = activation_type;
                this->input_shape = input_shape;
                this->outputShape = {input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, filters};
                this->bits = 32;
                this->use_bias = true;
                this->materialized = true;
                this->first_node = nullptr;
                this->kernel_size = kernel_size;
                this->filters = filters;
                this->for_output = for_output;
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
                return addNode(this->outputShape, {1, outputShape, 1}, NodeType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &outputShape, ActivationType activationType) {
                return addNode(this->outputShape, outputShape, NodeType::full, true, activationType);
            }

            shared_ptr<NNVertex> addOutput(const vector<size_t> &outputShape, const size_t outputKernelSize,
                                           NodeType nodeType, ActivationType activationType) {
                // todo: support other types of convolution nodes here
                auto result = addNode(outputShape[2], outputKernelSize, NodeType::convolution2dValid, true,
                                      activationType);
                if(result->outputShape[0] != outputShape[0] ||
                   result->outputShape[1] != outputShape[1] ||
                   result->outputShape[2] != outputShape[2]) {
                    stringstream ss;
                    ss << "The calculated output shape of the node ("
                       << result->outputShape[0] << ", " << result->outputShape[1] << ", " << result->outputShape[2]
                       << ") didn't match the desired output shape ("
                       << outputShape[0] << ", " << outputShape[1] << ", " << outputShape[2] << ")";

                    // todo: could we have taken other action here to avoid an error and reshape the output?
                    throw exception(ss.str().c_str());
                }
                return result;
            }

            shared_ptr<NNVertex> addNode(const size_t outputShape, NodeType nodeType, ActivationType activationType) {
                return addNode({1, outputShape, 1}, nodeType, activationType);
            }

            shared_ptr<NNVertex> addNode(const vector<size_t> &outputShape, NodeType nodeType,
                                         ActivationType activationType) {
                return addNode(this->outputShape, outputShape, nodeType, false, activationType);
            }

            shared_ptr<NNVertex> addNode(const size_t next_filters, const size_t next_kernel_size, NodeType nodeType,
                                         ActivationType activationType) {
                return addNode(next_filters, next_kernel_size, nodeType, false, activationType);
            }

            shared_ptr<NNVertex> addNode(const size_t next_filters, const size_t next_kernel_size,
                                         NodeType nodeType, bool next_for_output, ActivationType activationType) {
                auto nnv = make_shared<NNVertex>(parent.lock(), nodeType, this->outputShape,
                                                 next_filters, next_kernel_size, next_for_output, activationType);
                auto nne = make_shared<NNEdge>();
                nne->from = shared_from_this();
                nne->to = nnv;
                edges.push_back(nne);
                return nnv;
            }

            shared_ptr<NNVertex> addNode(const vector<size_t> &inputShape,
                                         const vector<size_t> &outputShape, NodeType nodeType,
                                         bool next_for_output, ActivationType activationType) {
                auto nnv = make_shared<NNVertex>(parent.lock(), nodeType, inputShape, outputShape, next_for_output, activationType);
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

            shared_ptr<NeuralNetworkNode> build_node(const shared_ptr<NeuralNetworkForTraining> &nn) {
                if (first_node) {
                    return first_node;
                }
                shared_ptr<Optimizer> optimizer = nn->getOptimizer();
                shared_ptr<NeuralNetworkNode> next_node;
                shared_ptr<NeuralNetworkNode> last_node = nullptr;
                if (node_type == NodeType::full) {
                    if(input_shape[0] > 1) {
                        auto flatten_node = make_shared<NeuralNetworkNode>(make_shared<NeuralNetworkFlattenFunction>());
                        last_node = appendNode(last_node, flatten_node);
                    }
                    next_node = make_shared<NeuralNetworkNode>(
                            optimizer->createFullyConnectedNeurons(input_shape[0]*input_shape[1]*input_shape[2], outputShape[0] * outputShape[1] * outputShape[2], bits));
                } else if(node_type == NodeType::convolution2dValid) {
                    next_node = make_shared<NeuralNetworkNode>(optimizer->createConvolutional2d(input_shape, filters,
                                                                                                kernel_size, bits));
                } else {
                    throw exception("Unimplemented NodeType");
                }

                last_node = appendNode(last_node, next_node);

                if (use_bias) {
                    auto bias_node = make_shared<NeuralNetworkNode>(optimizer->createBias(outputShape, outputShape, bits));
                    last_node = appendNode(last_node, bias_node);
                }

                shared_ptr<ActivationFunction> activationFunction = createActivationFunction();
                auto activation_node = make_shared<NeuralNetworkOutputNode>(
                        make_shared<NeuralNetworkActivationFunction>(activationFunction));
                last_node = appendNode(last_node, activation_node);

                if (for_output) {
                    nn->addOutput(activation_node);
                }

                last_node->setMaterialized(materialized);
                for (const auto &edge: edges) {
                    auto childNode = edge->to->build_node(nn);
                    last_node->add(childNode);
                }
                return first_node;
            }

            shared_ptr<ActivationFunction> createActivationFunction() const {
                shared_ptr<ActivationFunction> activationFunction;
                switch (activation_type) {
                    case tanh:
                        activationFunction = make_shared<TanhActivationFunction>();
                        break;
                    case relu:
                        activationFunction = make_shared<ReLUActivationFunction>();
                        break;
                    case sigmoid:
                        activationFunction = make_shared<SigmoidActivationFunction>();
                        break;
                    case sigmoid_approx:
                        activationFunction = make_shared<SigmoidApproximationActivationFunction>();
                        break;
                    case tanh_approx:
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

            shared_ptr<NeuralNetworkNode> appendNode(const shared_ptr<NeuralNetworkNode> &last_node, const shared_ptr<NeuralNetworkNode> &node) {
                if(!first_node) {
                    first_node = node;
                    return node;
                }
                return last_node->add(node);
            }

        private:
            weak_ptr<MicromlDSL> parent;
            vector<shared_ptr<NNEdge>> edges;
            NodeType node_type;
            vector<size_t> input_shape;
            vector<size_t> outputShape;
            ActivationType activation_type;
            bool materialized;
            bool use_bias;
            uint8_t bits;
            shared_ptr<NeuralNetworkNode> first_node;
            size_t kernel_size{};
            size_t filters{};
            bool for_output;
        };

        shared_ptr<NNVertex> addInput(const size_t input_shape, const size_t output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            return addInput({1, input_shape, 1}, {1, output_shape, 1}, nodeType, activationType);
        }

        shared_ptr<NNVertex> addInput(const vector<size_t> &input_shape,
                                      const vector<size_t> &output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape, output_shape, false, activationType);
            heads.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutput(const vector<size_t> &input_shape,
                                      const vector<size_t> &output_shape, NodeType nodeType,
                                      ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape, output_shape, true, activationType);
            heads.push_back(nnv);
            return nnv;
        }

        // where kernel_size is the width and height of the convolution window being applied to the input
        // filters is the same as the depth of the output
        shared_ptr<NNVertex> addInput(const vector<size_t> &input_shape,
                                      const size_t filters, const size_t kernel_size,
                                      NodeType nodeType,
                 ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape, filters, kernel_size, false, activationType);
            heads.push_back(nnv);
            return nnv;
        }

        shared_ptr<NNVertex> addInputOutput(const vector<size_t> &input_shape,
                                            const size_t filters, const size_t kernel_size,
                                            NodeType nodeType, ActivationType activationType) {
            auto nnv = make_shared<NNVertex>(shared_from_this(), nodeType, input_shape, filters, kernel_size, true, activationType);
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
        float bias_learning_rate;
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
