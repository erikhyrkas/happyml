//
// Created by Erik Hyrkas on 11/23/2022.
//

#ifndef MICROML_NEURAL_NETWORK_HPP
#define MICROML_NEURAL_NETWORK_HPP

#include <set>
#include <vector>
#include "loss.hpp"
#include "activation.hpp"
#include "neural_network_function.hpp"
#include "optimizer.hpp"
#include "../util/timers.hpp"
#include "../util/basic_profiler.hpp"
#include "../util/tensor_utils.hpp"
#include "../util/unit_test.hpp"
#include "../types/tensor.hpp"
#include "../training_data/training_dataset.hpp"

using namespace std;

namespace microml {

    // A node is a vertex in a graph
    class NeuralNetworkNode : public enable_shared_from_this<NeuralNetworkNode> {
    public:
        explicit NeuralNetworkNode(
                const shared_ptr<NeuralNetworkFunction> &neuralNetworkFunction) { //}, const shared_ptr<Optimizer> &optimizer) {
            this->neuralNetworkFunction = neuralNetworkFunction;
            //this->optimizer = optimizer;
            this->materialized = true;
        }

        virtual void sendOutput(shared_ptr<BaseTensor> &output) {
        }

        void setMaterialized(bool m) {
            materialized = m;
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
        void backward(shared_ptr<BaseTensor> &outputError) {
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

//        shared_ptr<NeuralNetworkNode> addFullyConnected(size_t input_size, size_t output_size, bool use_32_bit) {
//            auto networkFunction = optimizer->createFullyConnectedNeurons(input_size, output_size, use_32_bit);
//            auto networkNode = make_shared<NeuralNetworkNode>(networkFunction, optimizer);
//            return add(networkNode);
//        }
//
//        shared_ptr<NeuralNetworkNode> addActivation(const shared_ptr<ActivationFunction> &activationFunction) {
//            auto networkFunction = make_shared<NeuralNetworkActivationFunction>(activationFunction);
//            auto networkNode = make_shared<NeuralNetworkNode>(networkFunction, optimizer);
//            return add(networkNode);
//        }
//        shared_ptr<NeuralNetworkNode> addBias() {
//            auto networkFunction = make_shared<NeuralNetworkBias>(activationFunction);
//            auto networkNode = make_shared<NeuralNetworkNode>(networkFunction, optimizer);
//            return add(networkNode);
//        }

    private:
        vector<weak_ptr<NeuralNetworkConnection>> connectionInputs;
        vector<shared_ptr<NeuralNetworkConnection>> connectionOutputs;
        shared_ptr<NeuralNetworkFunction> neuralNetworkFunction;
        bool materialized;
//        shared_ptr<Optimizer> optimizer;
    };

    class NeuralNetworkOutputNode : public NeuralNetworkNode {
    public:
        explicit NeuralNetworkOutputNode(const shared_ptr<NeuralNetworkFunction> &neuralNetworkFunction)
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


    // TODO: this supports training and inference,
    // but we could load a Neural network for inference (prediction) that was lower overhead.
    //
    // You don't need an optimizer for predictions if you already have weights and you aren't going
    // to change those weights. Optimizers save extra state while doing predictions that
    // we wouldn't need to save if we are never going to use it.
    class NeuralNetwork {
    public:

        float predictScalar(const shared_ptr<BaseTensor> &givenInputs) {
            return scalar(predict(givenInputs)[0]);
        }

        shared_ptr<BaseTensor> predictOne(const shared_ptr<BaseTensor> &givenInputs) {
            return predict(givenInputs)[0];
        }

        vector<shared_ptr<BaseTensor>> predict(const shared_ptr<BaseTensor> &givenInputs) {
            return predict(vector<shared_ptr<BaseTensor>>{givenInputs});
        }

        vector<shared_ptr<BaseTensor>> predict(const vector<shared_ptr<BaseTensor>> &givenInputs) {
            return predict(givenInputs, false);
        }
        // predict/infer
        // I chose the word "predict" because it is more familiar than the word "infer"
        // and the meaning is more or less the same.
        vector<shared_ptr<BaseTensor>> predict(const vector<shared_ptr<BaseTensor>> &givenInputs, bool forTraining) {
            if (givenInputs.size() != headNodes.size()) {
                throw exception("infer requires as many input tensors as there are input nodes");
            }
            for (size_t i = 0; i < headNodes.size(); i++) {
                headNodes[i]->forwardFromInput(givenInputs[i], forTraining);
            }
            vector<shared_ptr<BaseTensor>> results;
            for (const auto &output: outputNodes) {
                results.push_back(output->consumeLastOutput());
            }

            return results;
        }

        void addHead(const shared_ptr<NeuralNetworkNode> &head) {
            headNodes.push_back(head);
        }

        void addOutput(const shared_ptr<NeuralNetworkOutputNode> &output) {
            outputNodes.push_back(output);
        }

    protected:
        vector<shared_ptr<NeuralNetworkNode>> headNodes;
        vector<shared_ptr<NeuralNetworkOutputNode>> outputNodes;
    };

    class NeuralNetworkForTraining : public NeuralNetwork {
    public:
        NeuralNetworkForTraining(const shared_ptr<LossFunction> &lossFunction, const shared_ptr<Optimizer> &optimizer) {
            this->lossFunction = lossFunction;
            this->optimizer = optimizer;
        }

        void setLossFunction(const shared_ptr<LossFunction> &f) {
            this->lossFunction = f;
        }

        shared_ptr<Optimizer> getOptimizer() {
            return optimizer;
        }

        // a sample is a single record
        // a batch is the number of samples (records) to look at before updating weights
        // train/fit
        void train(const shared_ptr<TrainingDataSet> &source, size_t epochs, int batchSize = 1,
                   bool overwriteOutputLines = true) {
            auto total_records = source->recordCount();
            if (batchSize > total_records) {
                throw exception("Batch Size cannot be larger than source data set.");
            }
            ElapsedTimer totalTimer;
            const size_t outputSize = outputNodes.size();
            cout << endl;
            logTraining(0, -1, epochs, 0, ceil(total_records / batchSize), batchSize, 0, overwriteOutputLines);
            for (size_t epoch = 0; epoch < epochs; epoch++) {
                ElapsedTimer timer;
                source->shuffle();

                int batchOffset = 0;
                float epochLoss = 0.f;
                vector<vector<shared_ptr<BaseTensor>>> batchPredictions;
                vector<vector<shared_ptr<BaseTensor>>> batchTruths;
                batchPredictions.resize(outputSize);
                batchTruths.resize(outputSize);

                size_t current_record = 0;
                auto nextRecord = source->nextRecord();
                while (nextRecord != nullptr) {
                    current_record++;
                    auto nextGiven = nextRecord->getGiven();
                    auto nextTruth = nextRecord->getExpected();
                    auto nextPrediction = predict(nextGiven, true);
                    for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {
                        batchPredictions[outputIndex].push_back(nextPrediction[outputIndex]);
                        batchTruths[outputIndex].push_back(nextTruth[outputIndex]);
//                        if(batch_predictions.size() != batch_truths.size()) {
//                            throw exception("truths and predictions should be equal");
//                        }
                    }
                    batchOffset++;
                    nextRecord = source->nextRecord();
                    if (batchOffset >= batchSize || nextRecord == nullptr) {
                        size_t currentBatch = ceil(current_record / batchSize);
                        double totalBatchOutputLoss = 0;
                        for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {
                            // TODO: materializing the error into a full tensor helps performance at the cost of memory.
                            //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                            //  to use for performance.
                            auto totalError = make_shared<FullTensor>(
                                    lossFunction->calculateTotalError(batchTruths[outputIndex],
                                                                      batchPredictions[outputIndex]));
                            auto totalLoss = lossFunction->compute(totalError);
                            auto batchLoss = totalLoss / (float)batchOffset;
                            totalBatchOutputLoss += batchLoss;

                            // batchOffset should be equal to batch_size, unless we are on the last partial batch.
                            auto lossDerivative = lossFunction->partialDerivative(totalError, (float) batchOffset);

                            // todo: we don't weight loss when there are multiple outputs back propagating. we should, instead of treating them as equals.
                            outputNodes[outputIndex]->backward(lossDerivative);

                            batchTruths[outputIndex].clear();
                            batchPredictions[outputIndex].clear();
                        }
                        // for each offset:
                        //   average = average + (val[offset] - average)/(offset+1)
                        // TODO: this loss assumes that all outputs have the same weight, which may not be true:
                        epochLoss += (float) (((totalBatchOutputLoss/(double)outputSize) - epochLoss) / (double)currentBatch);
                        auto elapsedTime = timer.getMilliseconds();
                        logTraining(elapsedTime, epoch, epochs, currentBatch,
                                    ceil(total_records / batchSize), batchOffset, epochLoss,
                                    overwriteOutputLines);
                        batchOffset = 0;
                    }
                }
                source->restart();
            }
            long long int elapsed = totalTimer.getMilliseconds();
            if (elapsed < 2000) {
                cout << endl << "Finished training in " << elapsed << " milliseconds." << endl;
            } else if (elapsed < 120000) {
                cout << endl << "Finished training in " << (elapsed / 1000) << " seconds." << endl;
            } else {
                cout << endl << "Finished training in " << (elapsed / 60000) << " minutes." << endl;
            }
        }

        static void logTraining(long long int elapsedTime, size_t epoch, size_t epochs,
                                size_t currentRecord, size_t totalRecords, int batchSize,
                                float loss, bool overwrite) {
            // printf is about 6x faster than cout. I'm not sure why, since neither should flush without an end line.
            // I can only assume it relates to how cout processes numbers to strings.

            if (elapsedTime > 120000) {
                auto min = elapsedTime / 60000;
                auto sec = (elapsedTime % 60000) / 1000;
                printf("%5zd m %zd s\tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f      ",
                       min, sec, (epoch + 1), epochs,
                       currentRecord, totalRecords, batchSize, loss);
            } else if (elapsedTime > 2000) {
                printf("%5zd s\tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f            ",
                       (elapsedTime / 1000), (epoch + 1), epochs,
                       currentRecord, totalRecords, batchSize, loss);
            } else {
                printf("%5zd ms\tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f           ",
                       elapsedTime, (epoch + 1), epochs,
                       currentRecord, totalRecords, batchSize, loss);
            }
            if (overwrite) {
                printf("\r");
            } else {
                printf("\n");
            }
        }

//        bool hasCycle() {
//            set<NeuralNetworkNode *> visited;
//            for (shared_ptr<NeuralNetworkNode> node: heads) {
//                if (node->hasCycle(visited)) {
//                    return true;
//                }
//            }
//            return false;
//        }

//        shared_ptr<NeuralNetworkNode> addFullyConnected(const shared_ptr<Optimizer> &optimizer, size_t input_size, size_t output_size, bool use_32_bit) {
//            auto networkNode = optimizer->createFullyConnectedNeurons(input_size, output_size, use_32_bit);
//            head_nodes.push_back(networkNode);
//            return networkNode;
//        }
//
//        shared_ptr<NeuralNetworkNode> addOutput(const shared_ptr<NeuralNetworkNode> &previous, const shared_ptr<ActivationFunction> &activationFunction) {
//            auto networkFunction = make_shared<NeuralNetworkActivationFunction>(activationFunction);
//            auto networkNode = make_shared<NeuralNetworkOutputNode>(networkFunction, optimizer);
//            output_nodes.push_back(output);
//            return previous->add(networkNode);
//        }
    private:
        shared_ptr<Optimizer> optimizer;
        shared_ptr<LossFunction> lossFunction;
    };

}
#endif //MICROML_NEURAL_NETWORK_HPP
