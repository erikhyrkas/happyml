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

        cout << "Test with binary cross entropy" << endl;
        auto neuralNetwork = neuralNetworkBuilder()
                ->setModelName("or_example")
                ->setModelRepo("../happyml_repo/models/")
                ->setLossFunction(LossType::binaryCrossEntropy)
                ->addInputLayer(orDataSource->getGivenShape(), 64, LayerType::full, ActivationType::tanhApprox)
                ->addLayer(32, LayerType::full, ActivationType::tanhApprox)
                ->addLayer(8, LayerType::full, ActivationType::tanhApprox)
                ->addOutputLayer(orDataSource->getExpectedShape(), ActivationType::sigmoid)->setUseBias(true)
                ->build();

//        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(orDataSource)->final_test_loss;

        cout << fixed << setprecision(2);
        cout << "Result loss: " << loss << endl;
        cout << "0 OR 0 = 0 Prediction: " << round(neuralNetwork->predictScalar(columnVector({0.f, 0.f}))) << endl;
        cout << "0 OR 1 = 1 Prediction: " << round(neuralNetwork->predictScalar(columnVector({0.f, 1.f}))) << endl;
        cout << "1 OR 0 = 1 Prediction: " << round(neuralNetwork->predictScalar(columnVector({1.f, 0.f}))) << endl;
        cout << "1 OR 1 = 1 Prediction: " << round(neuralNetwork->predictScalar(columnVector({1.f, 1.f}))) << endl;

        // testing save logic:
        neuralNetwork->saveWithOverwrite();
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("or_example",
                                                                "../happyml_repo/models/");
        float testLoss = loadedNeuralNetwork->test(orDataSource);
        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
        orDataSource->restart();
        float accuracy = loadedNeuralNetwork->compute_binary_accuracy(orDataSource);
        cout << fixed << setprecision(2) << "Result accuracy: " << accuracy << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}
