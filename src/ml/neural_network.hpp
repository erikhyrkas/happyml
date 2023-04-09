//
// Created by Erik Hyrkas on 11/23/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_NEURAL_NETWORK_HPP
#define HAPPYML_NEURAL_NETWORK_HPP

#include <set>
#include <vector>
#include <filesystem>
#include "loss.hpp"
#include "optimizer.hpp"
#include "enums.hpp"
#include "exit_strategy.hpp"
#include "optimizer_factory.hpp"
#include "neural_network_function.hpp"
#include "../util/basic_profiler.hpp"
#include "../util/tensor_utils.hpp"
#include "../util/timers.hpp"
#include "../util/unit_test.hpp"
#include "../util/file_writer.hpp"
#include "../types/tensor.hpp"
#include "../training_data/training_dataset.hpp"

using namespace std;
using namespace happyml;

namespace happyml {

    // A node is a vertex in a graph
    class NeuralNetworkNode : public enable_shared_from_this<NeuralNetworkNode> {
    public:
        explicit NeuralNetworkNode(
                const shared_ptr<NeuralNetworkFunction> &neuralNetworkFunction) {
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
        shared_ptr<NeuralNetworkFunction> neuralNetworkFunction;
        bool materialized;
        bool saved;
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
        NeuralNetwork(const string &name, const string &repoRootPath) {
            this->name = name;
            this->repoRootPath = repoRootPath;
        }

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
            results.reserve(outputNodes.size());
            for (const auto &output: outputNodes) {
                results.push_back(output->consumeLastOutput());
            }

            return results;
        }

        void addHeadNode(const shared_ptr<NeuralNetworkNode> &head) {
            headNodes.push_back(head);
        }

        void addOutput(const shared_ptr<NeuralNetworkOutputNode> &output) {
            outputNodes.push_back(output);
        }

    protected:
        string name;
        string repoRootPath;
        vector<shared_ptr<NeuralNetworkNode>> headNodes;
        vector<shared_ptr<NeuralNetworkOutputNode>> outputNodes;
    };

    class NeuralNetworkForTraining : public NeuralNetwork {
    public:
        NeuralNetworkForTraining(const string &name, const string &repoRootPath, const OptimizerType optimizerType,
                                 const float learningRate, const float biasLearningRate,
                                 const LossType lossType)
                : NeuralNetwork(name, repoRootPath) {
            this->optimizerType = optimizerType;
            this->lossType = lossType;
            this->learningRate = learningRate;
            this->biasLearningRate = biasLearningRate;
            this->optimizer = createOptimizer(optimizerType, learningRate, biasLearningRate);
            this->lossFunction = createLoss(lossType);

            useLowPrecisionExitStrategy();
        }

        void useLowPrecisionExitStrategy() {
            setExitStrategy(make_shared<DefaultExitStrategy>(10,
                                                             NINETY_DAYS_MS,
                                                             1000000,
                                                             0.001,
                                                             0.001,
                                                             2));
        }

        void useHighPrecisionExitStrategy() {
            setExitStrategy(make_shared<DefaultExitStrategy>(10,
                                                             NINETY_DAYS_MS,
                                                             1000000,
                                                             0.00001,
                                                             0.00001,
                                                             2));
        }

        void setExitStrategy(const shared_ptr<ExitStrategy> &updatedExitStrategy) {
            this->exitStrategy = updatedExitStrategy;
        }

        void setLossFunction(const shared_ptr<LossFunction> &f) {
            this->lossFunction = f;
        }

        shared_ptr<BaseOptimizer> getOptimizer() {
            return optimizer;
        }

        void saveWithOverwrite() {
            string modelPath = repoRootPath + "/" + name;
            saveAs(modelPath, true);
        }

        void saveWithoutOverwrite() {
            string modelPath = repoRootPath + "/" + name;
            saveAs(modelPath, false);
        }

        void saveAs(const string &modelFolderPath, bool overwrite) {
            auto modelPath = modelFolderPath;
            if (filesystem::is_directory(modelPath)) {
                if (!overwrite) {
                    // I don't want to throw an exception here since training a model can take a long time
                    // and people would be upset about losing their work, so we'll just save to a new location.
                    // Part of me thinks that it's better not to do this and just throw the error, and part of me
                    // thinks about how I have spent days waiting for some models to train and this sort of
                    // mistake would kill me.
                    auto canonicalModelPath = filesystem::canonical(modelPath);
                    cerr << "Model path " << canonicalModelPath
                         << " already existed, attempting to save to the new location: ";
                    auto ms = std::to_string(chrono::duration_cast<chrono::milliseconds>(
                            chrono::system_clock::now().time_since_epoch()).count());
                    canonicalModelPath += "_" + ms;
                    cerr << canonicalModelPath << endl;
                    modelPath = canonicalModelPath.generic_string();
                } else {
                    filesystem::remove_all(modelPath);
                }
            }
            filesystem::create_directories(modelPath);
            string modelProperties = modelPath + "/configuration.happyml";
            auto writer = make_unique<DelimitedTextFileWriter>(modelProperties, ':');
            for (const auto& record: networkMetadata) {
                writer->writeRecord(record);
            }
            writer->close();
            saveKnowledge(modelFolderPath, "default", overwrite);
        }

        void saveKnowledgeWithOverwrite(const string &knowledgeLabel) {
            saveKnowledge(knowledgeLabel, true);
        }

        void saveKnowledgeWithoutOverwrite(const string &knowledgeLabel) {
            saveKnowledge(knowledgeLabel, false);
        }

        void saveKnowledge(const string &knowledgeLabel, bool overwrite) {
            string modelPath = repoRootPath + "/" + name;
            saveKnowledge(modelPath, knowledgeLabel, overwrite);
        }

        void saveKnowledge(const string &modelFolderPath, const string &knowledgeLabel, bool overwrite) {
            string fullKnowledgePath = buildKnowledgePath(modelFolderPath, knowledgeLabel, overwrite);
            for (const auto & headNode : headNodes) {
                headNode->markUnsaved();
                headNode->saveKnowledge(fullKnowledgePath);
            }
        }

        void removeKnowledge(const string &knowledgeLabel) {
            string modelPath = repoRootPath + "/" + name;
            removeKnowledge(modelPath, knowledgeLabel);
        }

        static void removeKnowledge(const string &modelFolderPath, const string &knowledgeLabel) {
            string fullKnowledgePath = modelFolderPath + "/" + knowledgeLabel;
            filesystem::remove_all(fullKnowledgePath);
        }

        void loadKnowledge(const string &knowledgeLabel) {
            string modelPath = repoRootPath + "/" + name;
            loadKnowledge(modelPath, knowledgeLabel);
        }

        void loadKnowledge(const string &modelFolderPath, const string &knowledgeLabel) {
            string fullKnowledgePath = modelFolderPath + "/" + knowledgeLabel;
            for (const auto & headNode : headNodes) {
                headNode->markUnsaved();
                headNode->loadKnowledge(fullKnowledgePath);
            }
        }

        float train(const shared_ptr<TrainingDataSet> &trainingDataset,
                    int batchSize = 1,
                    TrainingRetentionPolicy trainingRetentionPolicy = best,
                    bool overwriteOutputLines = true) {
            auto testDataset = make_shared<EmptyTrainingDataSet>();
            return train(trainingDataset,
                         testDataset,
                         batchSize,
                         trainingRetentionPolicy,
                         overwriteOutputLines);
        }

        // a sample is a single record
        // a batch is the number of samples (records) to look at before updating weights
        // train/fit
        float train(const shared_ptr<TrainingDataSet> &trainingDataset,
                    const shared_ptr<TrainingDataSet> &testDataset,
                    int batchSize = 1,
                    TrainingRetentionPolicy trainingRetentionPolicy = best,
                    bool overwriteOutputLines = true) {
            string knowledgeCheckpointLabel = "checkpoint_" + std::to_string(
                    chrono::duration_cast<chrono::milliseconds>(
                            chrono::system_clock::now().time_since_epoch()).count());
            bool useTestDataset = testDataset->recordCount() > 0;
            // TODO: take in a test set and then validate that the records aren't in the training
            //  set. If they are, give a warning.
            auto total_records = trainingDataset->recordCount();
            if (batchSize > total_records) {
                throw exception("Batch Size cannot be larger than trainingDataset data set.");
            }
            ElapsedTimer totalTimer;
            const size_t outputSize = outputNodes.size();
            cout << endl;
            size_t lowestLossEpoch = 0;
            float lowestLoss = INFINITY;
            logTraining(0, 0, 0, ceil(total_records / batchSize), batchSize, 0, 0, 0, overwriteOutputLines);
            size_t epoch = 0;
            ElapsedTimer epochTimer;
            float epochTrainingLoss;
            float epochTestingLoss;
            do {
                ElapsedTimer batchTimer;
                trainingDataset->shuffle();

                optimizer->update_time_step();

                epochTrainingLoss = 0.f;
                int batchOffset = 0;
                vector<vector<shared_ptr<BaseTensor>>> batchPredictions;
                vector<vector<shared_ptr<BaseTensor>>> batchTruths;
                batchPredictions.resize(outputSize);
                batchTruths.resize(outputSize);

                size_t current_record = 0;
                auto nextRecord = trainingDataset->nextRecord();
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

                    nextRecord = trainingDataset->nextRecord();
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
                            auto batchLoss = totalLoss / (float) batchOffset;
                            totalBatchOutputLoss += batchLoss;

                            // batchOffset should be equal to batch_size, unless we are on the last partial batch.
                            shared_ptr<BaseTensor> lossDerivative = make_shared<TensorClipView>(lossFunction->partialDerivative(totalError, (float) batchOffset), 100, -100);
//                            shared_ptr<BaseTensor> lossDerivative = lossFunction->partialDerivative(totalError, (float) batchOffset);

                            // todo: we don't weight loss when there are multiple outputs back propagating. we should, instead of treating them as equals.
                            outputNodes[outputIndex]->backward(lossDerivative);
                            if(::isnan(batchLoss)) {
                                throw exception("Error calculating loss.");
                            }
                            batchTruths[outputIndex].clear();
                            batchPredictions[outputIndex].clear();
                        }
                        // for each offset:
                        //   average = average + (val[offset] - average)/(offset+1)
                        // TODO: this loss assumes that all outputs have the same weight, which may not be true:
                        epochTrainingLoss += (float) (
                                ((totalBatchOutputLoss / (double) outputSize) - epochTrainingLoss) /
                                (double) currentBatch);
                        auto elapsedTime = batchTimer.peekMilliseconds();
                        logTraining(elapsedTime, epoch, currentBatch,
                                    ceil(total_records / batchSize), batchOffset,
                                    epochTrainingLoss, lowestLoss, lowestLossEpoch,
                                    overwriteOutputLines);
                        batchOffset = 0;
                    }
                }
                if (overwriteOutputLines) {
                    cout << endl;
                }
                if (useTestDataset) {
                    epochTestingLoss = test(testDataset);
                } else {
                    epochTestingLoss = epochTrainingLoss;
                }
                if (epochTestingLoss < lowestLoss) {
                    lowestLoss = epochTestingLoss;
                    lowestLossEpoch = epoch;
                    if (trainingRetentionPolicy == best) {
                        saveKnowledge(knowledgeCheckpointLabel, true);
                    }
                    logTraining(batchTimer.peekMilliseconds(), epoch, batchSize,
                                ceil(total_records / batchSize), batchSize,
                                epochTrainingLoss, lowestLoss, lowestLossEpoch,
                                overwriteOutputLines);
                }
                trainingDataset->restart();
                epoch++;
            } while (!exitStrategy->isDone(epoch, epochTestingLoss, epochTimer.getMilliseconds()));
            int64_t elapsed = totalTimer.getMilliseconds();
            cout << endl << "Finished training in ";
            if (elapsed < 2000) {
                cout << elapsed << " milliseconds." << endl;
            } else if (elapsed < 120000) {
                cout << (elapsed / 1000) << " seconds." << endl;
            } else {
                cout << (elapsed / 60000) << " minutes." << endl;
            }
            // TODO: this is placeholder code until we actually save and formalize best loss,
            //  but it simulates the future results.
            if (trainingRetentionPolicy == best) {
                loadKnowledge(knowledgeCheckpointLabel);
                removeKnowledge(knowledgeCheckpointLabel);
                return lowestLoss;
            }
            return epochTestingLoss;
        }

        float test(const shared_ptr<TrainingDataSet> &testDataset,
                   bool overwriteOutputLines = true) {
            testDataset->restart();
            const size_t outputSize = outputNodes.size();
            float averageLoss = 0;
            ElapsedTimer trainingTimer;
            size_t currentRecord = 0;
            size_t totalRecords = testDataset->recordCount();
            auto nextRecord = testDataset->nextRecord();
            while (nextRecord) {
                const auto nextGiven = nextRecord->getGiven();
                auto nextTruth = nextRecord->getExpected();
                auto nextPrediction = predict(nextGiven, false);
                float totalLoss = 0;
                for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {
                    const auto error = make_shared<FullTensor>(
                            lossFunction->calculateError(nextTruth[outputIndex],
                                                         nextPrediction[outputIndex]));
                    const auto loss = lossFunction->compute(error);
                    totalLoss += loss;
                }
                currentRecord++;
                averageLoss += (float) (((totalLoss / (double) outputSize) - averageLoss) / (double) currentRecord);
                logTesting(trainingTimer.peekMilliseconds(),
                           currentRecord, totalRecords,
                           averageLoss, overwriteOutputLines);
                nextRecord = testDataset->nextRecord();
            }
            if (overwriteOutputLines) {
                printf("\n");
            }
            return averageLoss;
        }

        static void logTesting(int64_t elapsedTime,
                               size_t currentRecord, size_t totalRecords,
                               float loss, bool overwrite) {
            // printf is about 6x faster than cout. I'm not sure why, since neither should flush without an end line.
            // I can only assume it relates to how cout processes numbers to strings.
            if (elapsedTime > 120000) {
                const auto min = elapsedTime / 60000;
                const auto sec = (elapsedTime % 60000) / 1000;
                printf("%5zd m %zd s ", min, sec);
            } else if (elapsedTime > 2000) {
                const auto sec = (elapsedTime / 1000);
                printf("%5zd s ", sec);
            } else {
                printf("%5zd ms ", elapsedTime);
            }
            printf("\tTesting: %4zd/%zd \tAverage Loss: %11f         ", currentRecord, totalRecords, loss);
            if (overwrite) {
                printf("\r");
            } else {
                printf("\n");
            }
        }

        static void logTraining(int64_t elapsedTime, size_t epoch,
                                size_t currentRecord, size_t totalRecords, int batchSize,
                                float loss, float lowestLoss, size_t lowestLossEpoch, bool overwrite) {
            // printf is about 6x faster than cout. I'm not sure why, since neither should flush without an end line.
            // I can only assume it relates to how cout processes numbers to strings.
            if (elapsedTime > 120000) {
                const auto min = elapsedTime / 60000;
                const auto sec = (elapsedTime % 60000) / 1000;
                printf("%5zd m %zd s ", min, sec);
            } else if (elapsedTime > 2000) {
                const auto sec = (elapsedTime / 1000);
                printf("%5zd s ", sec);
            } else {
                printf("%5zd ms ", elapsedTime);
            }
            printf("\tEpoch: %6zd \tBatch: %4zd/%zd Batch Size: %3d \tLoss: %11f",
                   (epoch + 1), currentRecord, totalRecords, batchSize, loss);
            if (epoch > 0) {
                printf(" \tLowest: %11f (%zd)            ", lowestLoss, (lowestLossEpoch + 1));
            }
            if (overwrite) {
                printf("\r");
            } else {
                printf("\n");
            }
        }

//        bool hasCycle() {
//            set<NeuralNetworkNode *> visited;
//            for (shared_ptr<NeuralNetworkNode> node: inputReceptors) {
//                if (node->hasCycle(visited)) {
//                    return true;
//                }
//            }
//            return false;
//        }

//        shared_ptr<NeuralNetworkNode> addFullyConnected(const shared_ptr<NodeFactory> &optimizer, size_t input_size, size_t output_size, bool use_32_bit) {
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
        void setNetworkMetadata(const vector<vector<string>> &newNetworkMetadata) {
            networkMetadata = newNetworkMetadata;
        }

    private:
        float learningRate;
        float biasLearningRate;
        OptimizerType optimizerType;
        LossType lossType;
        shared_ptr<BaseOptimizer> optimizer;
        shared_ptr<LossFunction> lossFunction;
        shared_ptr<ExitStrategy> exitStrategy;
        vector<vector<string>> networkMetadata;
    };

}
#endif //HAPPYML_NEURAL_NETWORK_HPP
