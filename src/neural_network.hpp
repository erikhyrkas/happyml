//
// Created by ehyrk on 11/23/2022.
//

#ifndef MICROML_NEURAL_NETWORK_HPP
#define MICROML_NEURAL_NETWORK_HPP

#include <vector>
#include <set>
#include "tensor.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "test/unit_test.hpp"
#include "activation.hpp"
#include "neural_network_function.hpp"
#include "optimizer.hpp"
#include "test/basic_profiler.hpp"

using namespace std;

namespace microml {

    class ElapsedTimer {
    public:
        ElapsedTimer() {
            start_time = std::chrono::high_resolution_clock::now();
        }

        long long int getMicroseconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getMilliseconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getSeconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    };


    // A node is a vertex in a graph
    class NeuralNetworkNode : public std::enable_shared_from_this<NeuralNetworkNode> {
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
            if(materialized) {
                // TODO: materializing the output into a full tensor helps performance at the cost of memory.
                //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                //  to use for performance.
                // TODO: It could already be materialized. We should have the ability to check if this is unneccessary.
                input_to_next = make_shared<FullTensor>(input_to_next);
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
            return temp; //make_shared<FullTensor>(temp);
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

        float predict_scalar(const shared_ptr<BaseTensor> &given_inputs) {
            return scalar(predict(given_inputs)[0]);
        }

        shared_ptr<BaseTensor> predictOne(const shared_ptr<BaseTensor> &given_inputs) {
            return predict(given_inputs)[0];
        }

        vector<shared_ptr<BaseTensor>> predict(const shared_ptr<BaseTensor> &given_inputs) {
            return predict(vector<shared_ptr<BaseTensor>>{given_inputs});
        }

        // predict/infer
        // I chose the word "predict" because it is more familiar than the word "infer"
        // and the meaning is more or less the same.
        vector<shared_ptr<BaseTensor>> predict(const vector<shared_ptr<BaseTensor>> &given_inputs) {
            if (given_inputs.size() != head_nodes.size()) {
                throw exception("infer requires as many input tensors as there are input nodes");
            }
            for (size_t i = 0; i < head_nodes.size(); i++) {
                head_nodes[i]->forwardFromInput(given_inputs[i]);
            }
            vector<shared_ptr<BaseTensor>> results;
            for (const auto &output: output_nodes) {
                results.push_back(output->consumeLastOutput());
            }

            return results;
        }

        void addHead(const shared_ptr<NeuralNetworkNode> &head) {
            head_nodes.push_back(head);
        }

        void addOutput(const shared_ptr<NeuralNetworkOutputNode> &output) {
            output_nodes.push_back(output);
        }

    protected:
        vector<shared_ptr<NeuralNetworkNode>> head_nodes;
        vector<shared_ptr<NeuralNetworkOutputNode>> output_nodes;
    };

    class NeuralNetworkForTraining : public NeuralNetwork {
    public:
        NeuralNetworkForTraining(const shared_ptr<LossFunction> &lossFunction, const shared_ptr<Optimizer> &optimizer) {
            this->lossFunction = lossFunction;
            this->optimizer = optimizer;
        }

        void setLossFunction(const shared_ptr<LossFunction> &loss_function) {
            this->lossFunction = loss_function;
        }

        shared_ptr<Optimizer> getOptimizer() {
            return optimizer;
        }

        // a sample is a single record
        // a batch is the number of samples (records) to look at before updating weights
        // train/fit
        void train(const shared_ptr<TrainingDataSet> &source, size_t epochs, int batch_size=1, bool overwrite_output_lines = false) {
            auto total_records = source->record_count();
            if(batch_size > total_records) {
                throw exception("Batch Size cannot be larger than source data set.");
            }
            ElapsedTimer total_timer;
            const size_t output_size = output_nodes.size();
            cout << endl;
            log_training(0, -1, epochs, 0, ceil(total_records/batch_size), batch_size, 0, 0, overwrite_output_lines);
            for (size_t epoch = 0; epoch < epochs; epoch++) {
                ElapsedTimer timer;
                source->shuffle();

                int batch_offset = 0;
                vector<vector<shared_ptr<BaseTensor>>> batch_predictions;
                vector<vector<shared_ptr<BaseTensor>>> batch_truths;
                batch_predictions.resize(output_size);
                batch_truths.resize(output_size);

                size_t current_record = 0;
                auto next_record = source->next_record();
                while (next_record != nullptr) {
                    current_record++;
                    auto next_given = next_record->getGiven();
                    auto next_truth = next_record->getExpected();
                    auto next_prediction = predict(next_given);
                    for (size_t output_index = 0; output_index < output_size; output_index++) {
                        batch_predictions[output_index].push_back(next_prediction[output_index]);
                        batch_truths[output_index].push_back(next_truth[output_index]);
//                        if(batch_predictions.size() != batch_truths.size()) {
//                            throw exception("truths and predictions should be equal");
//                        }
                    }
                    batch_offset++;
                    next_record = source->next_record();
                    if(batch_offset >= batch_size || next_record == nullptr) {
                        for (size_t output_index = 0; output_index < output_size; output_index++) {
                            // TODO: materializing the error into a full tensor helps performance at the cost of memory.
                            //  we should be able to determine the best strategy at runtime. Sometimes, memory is too valuable
                            //  to use for performance.
                            auto total_error = make_shared<FullTensor>(lossFunction->calculateTotalError(batch_truths[output_index], batch_predictions[output_index]));
//                            auto total_error = lossFunction->calculateTotalError(batch_truths[output_index], batch_predictions[output_index]);
                            auto loss = lossFunction->compute(total_error);
                            // batch_offset should be equal to batch_size, unless we are out of records.
                            auto loss_derivative = lossFunction->partialDerivative(total_error, (float)batch_offset);

                            auto elapsed_time = timer.getMilliseconds();
                            log_training(elapsed_time, epoch, epochs, ceil(current_record/batch_size), ceil(total_records/batch_size), batch_offset, loss, 1, overwrite_output_lines);
                            // todo: we don't weight loss when there are multiple outputs back propagating. we should, instead of treating them as equals.
                            output_nodes[output_index]->backward(loss_derivative);

                            elapsed_time = timer.getMilliseconds();
                            log_training(elapsed_time, epoch, epochs, ceil(current_record/batch_size), ceil(total_records/batch_size), batch_offset, loss, 2, overwrite_output_lines);
                            batch_truths[output_index].clear();
                            batch_predictions[output_index].clear();
                        }
                        batch_offset = 0;
                    }
                }
                source->restart();
            }
            cout << endl << "Finished training in " << total_timer.getSeconds() << " seconds." << endl;
        }

        static void log_training(long long int elapsed_time, size_t epoch, size_t epochs,
                     size_t current_record, size_t total_records, int batch_size,
                     float loss, int stage, bool overwrite) {
            // printf is about 6x faster than cout. I'm not sure why, since neither should flush without an end line.
            // I can only assume it relates to how cout processes numbers to strings.
            string status_message;
            switch(stage) {
                case 0:
                    status_message = "to initialize";
                    break;
                case 1:
                    status_message = "to predict";
                    break;
                case 2:
                    status_message = "to learn";
                    break;
                default:
                    status_message = "unknown";
            }
            if( elapsed_time > 120000 ) {
                auto min = elapsed_time / 60000;
                auto sec = (elapsed_time % 60000) / 1000;
                printf("%5zd m %zd s %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f      ",
                       min, sec, status_message.c_str(), (epoch + 1), epochs,
                       current_record, total_records, batch_size, loss);
            } else if(elapsed_time > 2000) {
                printf("%5zd s %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f            ",
                       (elapsed_time/1000), status_message.c_str(), (epoch + 1), epochs,
                       current_record, total_records, batch_size, loss);
            } else {
                printf("%5zd ms %-13s \tEpoch: %6zd/%zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f           ",
                       elapsed_time, status_message.c_str(), (epoch + 1), epochs,
                       current_record, total_records, batch_size, loss);
            }
            if(overwrite) {
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
