//
// Created by Erik Hyrkas on 11/24/2022.
// Copyright 2022. Usable under MIT license.
//

#include <memory>
#include "../ml/happyml_dsl.hpp"

using namespace std;
using namespace happyml;

int main() {
    try {
        auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
        // given input, expected result
        xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
        xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

        cout << "Test with tanhActivation" << endl;
        auto neuralNetwork = neuralNetworkBuilder(OptimizerType::sgd)
                ->setModelName("xor_example")
                ->setModelRepo("../happyml_repo/models/")
                ->setLossFunction(LossType::mse)
                ->addInputLayer(xorDataSource->getGivenShape(), 3, LayerType::full, ActivationType::tanhApprox)
                ->addOutputLayer(xorDataSource->getExpectedShape(), ActivationType::tanhApprox)
                ->build();
        neuralNetwork->useHighPrecisionExitStrategy();
        // For 32-bit: Results are good enough at 500 epochs, gets better with more epochs.
        // For 16-bit: 500 epochs seems good enough
        // For 8-bit: 2000 epochs seems good enough
        float loss = neuralNetwork->train(xorDataSource)->final_test_loss;

        cout << fixed << setprecision(2);
        cout << "Result loss: " << loss << endl;
        cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
        cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
        cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
        cout << "1 xor 1 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;

        // testing save logic:
        neuralNetwork->saveWithOverwrite();
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("xor_example",
                                                                "../happyml_repo/models/");
        float testLoss = loadedNeuralNetwork->test(xorDataSource);
        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}