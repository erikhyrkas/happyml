//
// Created by Erik Hyrkas on 11/24/2022.
//

#include <memory>
#include "../ml/model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;

int main() {
    try {
        auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
        // given input, expected result
        xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
        xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

        cout << "Test with tanh" << endl;
        auto neuralNetwork = neuralNetworkBuilder()
                ->setModelRepo("../repo/")
                ->setModelName("xor_example")
                ->addInput(xorDataSource->getGivenShape(), 3, NodeType::full, ActivationType::tanh_approx)
                ->addOutput(xorDataSource->getExpectedShape(), ActivationType::tanh_approx)
                ->build();
        // For 32-bit: Results are good enough at 500 epochs, gets better with more epochs.
        // For 16-bit: 500 epochs seems good enough
        // For 8-bit: 2000 epochs seems good enough
        float loss = neuralNetwork->train(xorDataSource);

        // TODO: save is a work in progress.
        // neuralNetwork->save();

        cout << fixed << setprecision(2);
        cout << "Result loss: " << loss << endl;
        cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 0.f})) << endl;
        cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({0.f, 1.f})) << endl;
        cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 0.f})) << endl;
        cout << "1 xor 1 = 0 Prediction: " << neuralNetwork->predictScalar(columnVector({1.f, 1.f})) << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}