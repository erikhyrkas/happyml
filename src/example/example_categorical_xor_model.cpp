//
// Created by Erik Hyrkas on 5/7/2023.
// Copyright 2023. Usable under MIT license.
//
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../ml/happyml_dsl.hpp"

using namespace std;
using namespace happyml;

// Categorical Cross Entropy
int main() {
    try {
        auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
        // given input, expected result
        xorDataSource->addTrainingData(columnVector({0.f, 0.f}), columnVector({1.f, 0.f}));
        xorDataSource->addTrainingData(columnVector({0.f, 1.f}), columnVector({0.f, 1.f}));
        xorDataSource->addTrainingData(columnVector({1.f, 0.f}), columnVector({0.f, 1.f}));
        xorDataSource->addTrainingData(columnVector({1.f, 1.f}), columnVector({1.f, 0.f}));

        cout << "Test with categorical cross entropy" << endl;
        auto neuralNetwork = neuralNetworkBuilder(OptimizerType::adam)
                ->setModelName("cat_xor_example")
                ->setModelRepo("../repo/")
                ->setLossFunction(LossType::categoricalCrossEntropy)
                ->addInputLayer(xorDataSource->getGivenShape(), 32, LayerType::full, ActivationType::leaky)
                ->addLayer(16, LayerType::full, ActivationType::leaky)
                ->addLayer(8, LayerType::full, ActivationType::sigmoid)
                ->addOutputLayer(xorDataSource->getExpectedShape(), ActivationType::softmax)
                ->build();

        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(xorDataSource, 4);

        cout << fixed << setprecision(2);
        cout << "Result loss: " << loss << endl;
        cout << "0 XOR 0 = [1, 0] Prediction: ";
        auto value1 = neuralNetwork->predictOne(columnVector({0.f, 0.f}));
        cout << "[" << value1->getValue(0, 0, 0) << ", " << value1->getValue(0, 1, 0) << "]" << endl;

        cout << "0 XOR 1 = [0, 1] Prediction: ";
        auto value2 = neuralNetwork->predictOne(columnVector({0.f, 1.f}));
        cout << "[" << value2->getValue(0, 0, 0) << ", " << value2->getValue(0, 1, 0) << "]" << endl;

        cout << "1 XOR 0 = [0, 1] Prediction: ";
        auto value3 = neuralNetwork->predictOne(columnVector({1.f, 0.f}));
        cout << "[" << value3->getValue(0, 0, 0) << ", " << value3->getValue(0, 1, 0) << "]" << endl;

        cout << "1 XOR 1 = [1, 0] Prediction: ";
        auto value4 = neuralNetwork->predictOne(columnVector({1.f, 1.f}));
        cout << "[" << value4->getValue(0, 0, 0) << ", " << value4->getValue(0, 1, 0) << "]" << endl;

        // testing save logic:
        neuralNetwork->saveWithOverwrite();
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("cat_xor_example",
                                                                "../repo/");
        float testLoss = loadedNeuralNetwork->test(xorDataSource);
        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}
