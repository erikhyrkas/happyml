//
// Created by Erik Hyrkas on 11/24/2022.
//

#include <memory>
#include "../ml/model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;


void usingTanh() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

    cout << "Test with tanh" << endl;
    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(xorDataSource->getGivenShape(), 3, NodeType::full, ActivationType::tanh_approx)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::tanh_approx)
            ->build();
    // For 32-bit: Results are good enough at 500 epochs, gets better with more epochs.
    // For 16-bit: 500 epochs seems good enough
    // For 8-bit: 2000 epochs seems good enough
    neuralNetwork->train(xorDataSource, 500, 1, true);

    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;
}

void usingSigmoid() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

    cout << "Test with sigmoid" << endl;
    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(xorDataSource->getGivenShape(), 5, NodeType::full, ActivationType::tanh)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::sigmoid)
            ->build();

    // good enough results if you round to 0 or 1:
    // 32-bit input node: input size 4,    1000 epochs (8 + 4x4x2  + 8 = 48 bytes)
    // 16-bit input node: input size 4,    1000 epochs (8 + 2x4x2  + 8 = 32 bytes)
    //  8-bit input node: input size 32!!, 1500 epochs (8 + 1x32x2 + 8 = 80 bytes)
    // Clearly, 8-bit doesn't work well for memory savings or quality results in this case.
    neuralNetwork->train(xorDataSource, 1000, 1, true);

    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;
}


void usingRelu() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

    cout << "Test with relu" << endl;
    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(xorDataSource->getGivenShape(), 5, NodeType::full, ActivationType::relu)->setBits(8)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::tanh)
            ->build();

    neuralNetwork->train(xorDataSource, 1000, 1, true);

    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;
}

int main() {
    try {
        // For this problem, tanh works far-and-away the best.
        // We can use other activation functions for the same problem, but they'll be less efficient at finding a result.
        // We don't particularly care about efficiency for this exact problem, only that activation functions work.
        // It is a good reminder, though, that picking the correct activation functions can dramatically improve
        // results and the time to train.
        usingTanh();
        usingSigmoid();
        usingRelu();

    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}