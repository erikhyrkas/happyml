//
// Created by ehyrk on 11/23/2022.
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
        void doForward(const vector<shared_ptr<BaseTensor>> &inputs) {
            auto input_to_next = neuralNetworkFunction->forward(inputs);
            if (materialized) {
                // TODO: materializing the output into a full tensor helps performance at the cost of memory.
                //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                //  to use for performance.
                input_to_next = materializeTensor(input_to_next);
            }
            if (connection_outputs.empty()) {
                // there are no nodes after this one, so we return our result.
                sendOutput(input_to_next);
                return;
            }
            for (const auto &output_connection: connection_outputs) {
                output_connection->next_input = input_to_next;
                output_connection->to->forwardFromConnection();
            }
        }

        void forwardFromInput(const shared_ptr<BaseTensor> &input) {
            return doForward({input});
        }

        void forwardFromConnection() {
            vector<shared_ptr<BaseTensor>> inputs;
            for (const auto &input: connection_inputs) {
                if (input.lock()->next_input == nullptr) {
                    return; // a different branch will populate the rest of the inputs, and we'll proceed then.
                }
                inputs.push_back(input.lock()->next_input);
            }
            doForward(inputs);
            for (const auto &input: connection_inputs) {
                input.lock()->next_input = nullptr;
            }
        }

        // todo: right now, i'm assuming this is a directed acyclic graph, this may not work for everything.
        //  It is possible, I'll have to track visited nodes to avoid infinite cycles.
        void backward(shared_ptr<BaseTensor> &output_error) {
            // TODO: for multiple errors, I'm currently averaging the errors as they propagate, but it probably should be a weighted average
//            cout <<endl << "Backward output error: " <<endl;
//            output_error->print();
//            cout <<endl;
            PROFILE_BLOCK(profileBlock);

            auto prior_error = neuralNetworkFunction->backward(output_error);
            if (materialized) {
                prior_error = materializeTensor(prior_error);
            }
            for (const auto &input_connection: connection_inputs) {
                PROFILE_BLOCK(backwardBlockLoop);
                const auto conn = input_connection.lock();
                const auto from = conn->from.lock();
                const auto from_connection_output_size = from->connection_outputs.size();
                if (from_connection_output_size == 1) {
                    PROFILE_BLOCK(backwardBlock);
                    // most of the time there is only one from, so, ship it instead of doing extra wasted calculations
                    from->backward(prior_error);
                } else {
                    PROFILE_BLOCK(backwardBlock);
                    // We'll save the error we calculated, because we need to sum the errors from all outputs
                    // and not all outputs may be ready yet.
                    conn->prior_error = prior_error;
                    bool ready = true;
                    shared_ptr<BaseTensor> sum = nullptr;
                    for (const auto &output_conn: from->connection_outputs) {
                        if (output_conn->prior_error == nullptr) {
                            ready = false;
                            break;
                        }
                        if (sum == nullptr) {
                            sum = make_shared<TensorAddTensorView>(prior_error, output_conn->prior_error);
                        } else {
                            sum = make_shared<TensorAddTensorView>(sum, output_conn->prior_error);
                        }
                    }
                    if (!ready) {
                        continue;
                    }
                    shared_ptr<BaseTensor> average_error = make_shared<TensorMultiplyByScalarView>(sum, 1.0f /
                                                                                                        (float) from_connection_output_size);
                    from->backward(average_error);
                    for (const auto &output_conn: from->connection_outputs) {
                        output_conn->prior_error = nullptr;
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
            shared_ptr<BaseTensor> prior_error;
            weak_ptr<NeuralNetworkNode> from;
            shared_ptr<NeuralNetworkNode> to;
        };


        shared_ptr<NeuralNetworkNode> add(const shared_ptr<NeuralNetworkNode> &child) {
            auto connection = make_shared<NeuralNetworkConnection>();
            // Avoid memory leaks created by circular strong reference chains.
            // We strongly own objects from the start of the graph toward the end,
            // rather than the end toward the start.
            connection->to = child; // strong reference to child
            connection_outputs.push_back(connection); // strong reference to connection to child
            connection->from = shared_from_this(); // weak reference to parent
            child->connection_inputs.push_back(connection); // weak reference to connection from parent
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
        vector<weak_ptr<NeuralNetworkConnection>> connection_inputs;
        vector<shared_ptr<NeuralNetworkConnection>> connection_outputs;
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

        // predict/infer
        // I chose the word "predict" because it is more familiar than the word "infer"
        // and the meaning is more or less the same.
        vector<shared_ptr<BaseTensor>> predict(const vector<shared_ptr<BaseTensor>> &givenInputs) {
            if (givenInputs.size() != headNodes.size()) {
                throw exception("infer requires as many input tensors as there are input nodes");
            }
            for (size_t i = 0; i < headNodes.size(); i++) {
                headNodes[i]->forwardFromInput(givenInputs[i]);
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
                   bool overwriteOutputLines = false) {
            auto total_records = source->recordCount();
            if (batchSize > total_records) {
                throw exception("Batch Size cannot be larger than source data set.");
            }
            ElapsedTimer totalTimer;
            const size_t outputSize = outputNodes.size();
            cout << endl;
            logTraining(0, -1, epochs, 0, ceil(total_records / batchSize), batchSize, 0, 0, overwriteOutputLines);
            for (size_t epoch = 0; epoch < epochs; epoch++) {
                ElapsedTimer timer;
                source->shuffle();

                int batchOffset = 0;
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
                    auto nextPrediction = predict(nextGiven);
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
                        for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {
                            // TODO: materializing the error into a full tensor helps performance at the cost of memory.
                            //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                            //  to use for performance.
                            auto totalError = make_shared<FullTensor>(
                                    lossFunction->calculateTotalError(batchTruths[outputIndex],
                                                                      batchPredictions[outputIndex]));
//                            auto total_error = lossFunction->calculateTotalError(batch_truths[output_index], batch_predictions[output_index]);
                            auto loss = lossFunction->compute(totalError);
                            // batch_offset should be equal to batch_size, unless we are out of records.
                            auto lossDerivative = lossFunction->partialDerivative(totalError, (float) batchOffset);

                            auto elapsedTime = timer.getMilliseconds();
                            logTraining(elapsedTime, epoch, epochs, ceil(current_record / batchSize),
                                        ceil(total_records / batchSize), batchOffset, loss, 1,
                                        overwriteOutputLines);
                            // todo: we don't weight loss when there are multiple outputs back propagating. we should, instead of treating them as equals.
                            outputNodes[outputIndex]->backward(lossDerivative);

                            elapsedTime = timer.getMilliseconds();
                            logTraining(elapsedTime, epoch, epochs, ceil(current_record / batchSize),
                                        ceil(total_records / batchSize), batchOffset, loss, 2,
                                        overwriteOutputLines);
                            batchTruths[outputIndex].clear();
                            batchPredictions[outputIndex].clear();
                        }
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
                                float loss, int stage, bool overwrite) {
            // printf is about 6x faster than cout. I'm not sure why, since neither should flush without an end line.
            // I can only assume it relates to how cout processes numbers to strings.
            string statusMessage;
            switch (stage) {
                case 0:
                    statusMessage = "to initialize";
                    break;
                case 1:
                    statusMessage = "to predict";
                    break;
                case 2:
                    statusMessage = "to learn";
                    break;
                default:
                    statusMessage = "unknown";
            }
            if (elapsedTime > 120000) {
                auto min = elapsedTime / 60000;
                auto sec = (elapsedTime % 60000) / 1000;
                printf("%5zd m %zd s %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f      ",
                       min, sec, statusMessage.c_str(), (epoch + 1), epochs,
                       currentRecord, totalRecords, batchSize, loss);
            } else if (elapsedTime > 2000) {
                printf("%5zd s %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f            ",
                       (elapsedTime / 1000), statusMessage.c_str(), (epoch + 1), epochs,
                       currentRecord, totalRecords, batchSize, loss);
            } else {
                printf("%5zd ms %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f           ",
                       elapsedTime, statusMessage.c_str(), (epoch + 1), epochs,
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
