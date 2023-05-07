//
// Created by Erik Hyrkas on 11/23/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_NEURAL_NETWORK_NODE_HPP
#define HAPPYML_NEURAL_NETWORK_NODE_HPP

#include <vector>
#include <filesystem>
#include "loss.hpp"
#include "optimizer.hpp"
#include "enums.hpp"
#include "exit_strategy.hpp"
#include "optimizer_factory.hpp"
#include "neural_network_layer_function.hpp"
#include "../util/timers.hpp"
#include "../util/unit_test.hpp"
#include "../util/file_writer.hpp"
#include "../training_data/training_dataset.hpp"
#include "../types/tensor_views/tensor_clip_view.hpp"

using namespace std;
using namespace happyml;

namespace happyml {

    // A node is a vertex in a graph, and most of the neural network nodes are "layers."
    // I was resistant to calling them layers, but I eventually gave in, because
    // it is the term that is most commonly used in the field.
    class NeuralNetworkNode : public enable_shared_from_this<NeuralNetworkNode> {
    public:
        explicit NeuralNetworkNode(
                const shared_ptr<NeuralNetworkLayerFunction> &neuralNetworkFunction) {
            this->neuralNetworkFunction = neuralNetworkFunction;
            this->materialized = true;
            this->saved = true;
        }

        virtual void sendOutput(shared_ptr<BaseTensor> &output) {
        }

        void setMaterialized(bool m) {
            materialized = m;
        }

        void markUnsaved() {
            if (saved) {
                saved = false;
                for (const auto &outputConnection: connectionOutputs) {
                    outputConnection->to->markUnsaved();
                }
            }
        }

        void saveKnowledge(const string &fullKnowledgePath) {
            if (saved) {
                return;
            }
            neuralNetworkFunction->saveKnowledge(fullKnowledgePath);
            for (const auto &outputConnection: connectionOutputs) {
                outputConnection->to->saveKnowledge(fullKnowledgePath);
            }
        }

        void loadKnowledge(const string &fullKnowledgePath) {
            if (saved) {
                return;
            }
            saved = true;
            neuralNetworkFunction->loadKnowledge(fullKnowledgePath);
            for (const auto &outputConnection: connectionOutputs) {
                outputConnection->to->loadKnowledge(fullKnowledgePath);
            }
        }

        // todo: right now, i'm assuming this is a directed acyclic graph, this may not work for everything.
        //  It is possible, I'll have to track visited nodes to avoid infinite cycles.
        void doForward(const vector<shared_ptr<BaseTensor>> &inputs, bool forTraining) {
            auto input_to_next = neuralNetworkFunction->forward(inputs, forTraining);
            if (materialized) {
                // TODO: materializing the output into a full tensor helps performance at the cost of memory.
                //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                //  to use for performance.
                input_to_next = materializeTensor(input_to_next);
            }
            if (connectionOutputs.empty()) {
                // there are no nodes after this one, so we return our result.
                sendOutput(input_to_next);
                return;
            }
            for (const auto &output_connection: connectionOutputs) {
                output_connection->next_input = input_to_next;
                output_connection->to->forwardFromConnection(forTraining);
            }
        }

        void forwardFromInput(const shared_ptr<BaseTensor> &input, bool forTraining) {
            return doForward({input}, forTraining);
        }

        void forwardFromConnection(bool forTraining) {
            vector<shared_ptr<BaseTensor>> inputs;
            for (const auto &input: connectionInputs) {
                auto lockedInput = input.lock();
                if (lockedInput->next_input == nullptr) {
                    return; // a different branch will populate the rest of the inputs, and we'll proceed then.
                }
                inputs.push_back(lockedInput->next_input);
            }
            doForward(inputs, forTraining);
            for (const auto &input: connectionInputs) {
                input.lock()->next_input = nullptr;
            }
        }

        // todo: right now, i'm assuming this is a directed acyclic graph, this may not work for everything.
        //  It is possible, I'll have to track visited nodes to avoid infinite cycles.
        void backward(const shared_ptr<BaseTensor> &outputError) {
            // TODO: for multiple errors, I'm currently averaging the errors as they propagate, but it probably should be a weighted average
            PROFILE_BLOCK(profileBlock);

            auto priorError = neuralNetworkFunction->backward(outputError);
            if (materialized) {
                priorError = materializeTensor(priorError);
            }
            for (const auto &inputConnection: connectionInputs) {
                PROFILE_BLOCK(backwardBlockLoop);
                const auto conn = inputConnection.lock();
                const auto from = conn->from.lock();
                const auto fromConnectionOutputSize = from->connectionOutputs.size();
                if (fromConnectionOutputSize == 1) {
                    PROFILE_BLOCK(backwardBlock);
                    // most of the time there is only one from, so, ship it instead of doing extra wasted calculations
                    from->backward(priorError);
                } else {
                    PROFILE_BLOCK(backwardBlock);
                    // We'll save the error we calculated, because we need to sum the errors from all outputs
                    // and not all outputs may be ready yet.
                    conn->priorError = priorError;
                    bool ready = true;
                    shared_ptr<BaseTensor> sum = nullptr;
                    for (const auto &output_conn: from->connectionOutputs) {
                        if (output_conn->priorError == nullptr) {
                            ready = false;
                            break;
                        }
                        if (sum == nullptr) {
                            sum = make_shared<TensorAddTensorView>(priorError, output_conn->priorError);
                        } else {
                            sum = make_shared<TensorAddTensorView>(sum, output_conn->priorError);
                        }
                    }
                    if (!ready) {
                        continue;
                    }
                    shared_ptr<BaseTensor> average_error = make_shared<TensorMultiplyByScalarView>(sum, 1.0f /
                                                                                                        (float) fromConnectionOutputSize);
                    from->backward(average_error);
                    for (const auto &output_conn: from->connectionOutputs) {
                        output_conn->priorError = nullptr;

                    }
                }
            }
            saved = false;
        }

//        bool hasCycle(set<NeuralNetworkNode *> &visited) {
//            if (visited.count(this) > 0) {
//                return true;
//            }
//            visited.insert(this);
//            for (shared_ptr<NeuralNetworkNode> c: children) {
//                if (c->hasCycle(visited)) {
//                    return true;
//                }
//            }
//            visited.erase(this);
//            return false;
//        }

        // A connection is also known as an "edge" in a graph, but not everybody remembers technical terms
        struct NeuralNetworkConnection {
            shared_ptr<BaseTensor> next_input;
            shared_ptr<BaseTensor> priorError;
            weak_ptr<NeuralNetworkNode> from;
            shared_ptr<NeuralNetworkNode> to;
        };


        shared_ptr<NeuralNetworkNode> add(const shared_ptr<NeuralNetworkNode> &child) {
            auto connection = make_shared<NeuralNetworkConnection>();
            // Avoid memory leaks created by circular strong reference chains.
            // We strongly own objects from the start of the graph toward the end,
            // rather than the end toward the start.
            connection->to = child; // strong reference to child
            connectionOutputs.push_back(connection); // strong reference to connection to child
            connection->from = shared_from_this(); // weak reference to parent
            child->connectionInputs.push_back(connection); // weak reference to connection from parent
            connection->next_input = nullptr;
            return child; // this lets us chain together calls in a builder format.
        }

    private:
        vector<weak_ptr<NeuralNetworkConnection>> connectionInputs;
        vector<shared_ptr<NeuralNetworkConnection>> connectionOutputs;
        shared_ptr<NeuralNetworkLayerFunction> neuralNetworkFunction;
        bool materialized;
        bool saved;
    };

    class NeuralNetworkOutputNode : public NeuralNetworkNode {
    public:
        explicit NeuralNetworkOutputNode(const shared_ptr<NeuralNetworkLayerFunction> &neuralNetworkFunction)
                : NeuralNetworkNode(neuralNetworkFunction) {
        }

        void sendOutput(shared_ptr<BaseTensor> &output) override {
            lastOutput = output;
        }

        shared_ptr<BaseTensor> consumeLastOutput() {
            auto temp = lastOutput;
            lastOutput = nullptr;
            // Always materialize output for performance of consumption or back propagation.
            // TODO: should we pick the precision of the materialization? right now, it is always 32-bit
            //  which might be excessive in some cases.
            return materializeTensor(temp);
        }

    private:
        shared_ptr<BaseTensor> lastOutput;
    };

}

#endif //HAPPYML_NEURAL_NETWORK_NODE_HPP
