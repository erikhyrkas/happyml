//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_NEURAL_NETWORK_HPP
#define HAPPYML_NEURAL_NETWORK_HPP

#include "neural_network_node.hpp"
#include "losses/mse_loss.hpp"
#include "losses/categorical_cross_entropy_loss.hpp"
#include "losses/binary_cross_entropy.hpp"
#include "losses/mae_loss.hpp"
#include "losses/smae_loss.hpp"

namespace happyml {

// You don't need an optimizer for predictions if you already have weights, and you aren't going
// to change those weights. Optimizers save extra state while doing predictions that
// we wouldn't need to save if we are never going to use it.
    class NeuralNetworkForPrediction {
    public:
        NeuralNetworkForPrediction(const string &name, const string &repoRootPath) {
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

    class NeuralNetworkForTraining : public NeuralNetworkForPrediction {
    public:
        NeuralNetworkForTraining(const string &name, const string &repoRootPath, const OptimizerType optimizerType,
                                 const float learningRate, const float biasLearningRate,
                                 const LossType lossType)
                : NeuralNetworkForPrediction(name, repoRootPath) {
            NeuralNetworkForTraining::optimizerType = optimizerType;
            NeuralNetworkForTraining::lossType = lossType;
            NeuralNetworkForTraining::learningRate = learningRate;
            NeuralNetworkForTraining::biasLearningRate = biasLearningRate;
            optimizer = createOptimizer(optimizerType, learningRate, biasLearningRate);
            switch (lossType) {
                case mse:
                    NeuralNetworkForTraining::lossFunction = make_shared<MeanSquaredErrorLossFunction>();
                    break;
                case mae:
                    NeuralNetworkForTraining::lossFunction = make_shared<MeanAbsoluteErrorLossFunction>();
                    break;
                case smae:
                    NeuralNetworkForTraining::lossFunction = make_shared<SmoothMeanAbsoluteErrorLossFunction>();
                    break;
                case categoricalCrossEntropy:
                    NeuralNetworkForTraining::lossFunction = make_shared<CategoricalCrossEntropyLossFunction>();
                    break;
                case binaryCrossEntropy:
                    NeuralNetworkForTraining::lossFunction = make_shared<BinaryCrossEntropyLossFunction>();
                    break;
                default:
                    NeuralNetworkForTraining::lossFunction = make_shared<MeanSquaredErrorLossFunction>();
            }

            useLowPrecisionExitStrategy();
        }

        void useLowPrecisionExitStrategy() {
            setExitStrategy(make_shared<DefaultExitStrategy>(10,
                                                             NINETY_DAYS_MS,
                                                             1000000,
                                                             0.001f,
                                                             0.001f,
                                                             2));
        }

        void useHighPrecisionExitStrategy() {
            setExitStrategy(make_shared<DefaultExitStrategy>(10,
                                                             NINETY_DAYS_MS,
                                                             1000000,
                                                             0.00001f,
                                                             0.00001f,
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
            string modelProperties = modelPath + "/configuration.hmlprops";
            auto writer = make_unique<DelimitedTextFileWriter>(modelProperties, ':');
            for (const auto &record: networkMetadata) {
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
            string fullKnowledgePath = initialize_knowledge_path_directory(modelFolderPath, knowledgeLabel, overwrite);
            for (const auto &headNode: headNodes) {
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
            for (const auto &headNode: headNodes) {
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
        // todo: we need support multiple for inputs. Now that we have the shuffler, we have
        //  a way to properly manage those multiple inputs, but this train method only has a single
        //  trainingDataset and testDataset for input.
        float train(const shared_ptr<TrainingDataSet> &trainingDataset,
                    const shared_ptr<TrainingDataSet> &testDataset,
                    int batchSize = 1,
                    TrainingRetentionPolicy trainingRetentionPolicy = best,
                    bool overwriteOutputLines = true) {
            ElapsedTimer totalTimer;
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

            auto trainingShuffler = make_shared<Shuffler>(trainingDataset->recordCount());
            trainingDataset->setShuffler(trainingShuffler);

            const size_t outputSize = outputNodes.size();
            size_t lowestLossEpoch = 0;
            float lowestLoss = INFINITY;
            cout << endl; // TODO: we really should have a silent mode.
            logTraining(0, 0, 0, ceil(total_records / batchSize), batchSize, 0, 0, 0, overwriteOutputLines);
            size_t epoch = 0;
            ElapsedTimer epochTimer;
            float epochTrainingLoss;
            float epochTestingLoss;
            do {
                ElapsedTimer batchTimer;
                trainingShuffler->shuffle();
                trainingDataset->restart();

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
                    }
                    batchOffset++;

                    nextRecord = trainingDataset->nextRecord();
                    if (batchOffset >= batchSize || nextRecord == nullptr) {
                        size_t currentBatch = ceil(current_record / batchSize);
                        double totalBatchOutputLoss = 0;
                        for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {
                            auto error_loss_derivative_pair = lossFunction->calculateBatchErrorAndDerivative(batchTruths[outputIndex],
                                                                                                             batchPredictions[outputIndex]);
                            auto batchError = error_loss_derivative_pair.first;
                            auto batchLoss = lossFunction->compute(batchError);
                            totalBatchOutputLoss += batchLoss;

                            auto lossDerivative = make_shared<TensorClipView>(error_loss_derivative_pair.second,
                                                                              -100.0f,
                                                                              100.0f);
                            // todo: we don't weight loss when there are multiple outputs back propagating. we should, instead of treating them as equals.
                            outputNodes[outputIndex]->backward(lossDerivative);
                            if (isnan(batchLoss)) {
                                throw exception("Error calculating loss.");
                            }
                            batchTruths[outputIndex].clear();
                            batchPredictions[outputIndex].clear();
                        }

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
                    shared_ptr<BaseTensor> error = make_shared<FullTensor>(
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
