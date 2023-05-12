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

        cout << "Test with reluActivation" << endl;
        auto neuralNetwork = neuralNetworkBuilder()
                ->addInputLayer(xorDataSource->getGivenShape(), 8, LayerType::full, ActivationType::relu)->setBits(8)
                ->addLayer(8, LayerType::full, ActivationType::relu)->setBits(8)
                ->addOutputLayer(xorDataSource->getExpectedShape(), ActivationType::tanhDefault)
                ->build();
        float loss = neuralNetwork->train(xorDataSource);

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