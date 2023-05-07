//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//
#include <memory>
#include "../ml/happyml_dsl.hpp"

using namespace std;
using namespace happyml;

// Binary Cross Entropy
int main() {
    try {
        auto orDataSource = make_shared<InMemoryTrainingDataSet>();
        // given input, expected result
        orDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
        orDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
        orDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
        orDataSource->addTrainingData(columnVector({1.f, 1.f}), 1.f);

        cout << "Test with binaryCrossEntropy" << endl;
        auto neuralNetwork = neuralNetworkBuilder(OptimizerType::microbatch)
                ->setModelName("or_example")
                ->setModelRepo("../repo/")
                ->setLossFunction(LossType::binaryCrossEntropy)
                ->addInput(orDataSource->getGivenShape(), 64, NodeType::full, ActivationType::tanhApprox)
                ->addNode(32, NodeType::full, ActivationType::tanhApprox)
                ->addNode(8, NodeType::full, ActivationType::tanhApprox)
                ->addOutput(orDataSource->getExpectedShape(), ActivationType::sigmoid)
                ->build();

//        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(orDataSource);

        cout << fixed << setprecision(2);
        cout << "Result loss: " << loss << endl;
        cout << "0 OR 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
        cout << "0 OR 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
        cout << "1 OR 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
        cout << "1 OR 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;

        // testing save logic:
        neuralNetwork->saveWithOverwrite();
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("or_example",
                                                                "../repo/");
        float testLoss = loadedNeuralNetwork->test(orDataSource);
        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}
