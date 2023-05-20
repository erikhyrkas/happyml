//
// Created by Erik Hyrkas on 11/24/2022.
// Copyright 2022. Usable under MIT license.
//

#include <memory>
#include "../ml/happyml_dsl.hpp"

using namespace std;
using namespace happyml;
using namespace happyml;

int main() {
    try {
        auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
        // given input, expected result
        xorDataSource->addTrainingData(columnVector({0.f, 0.f}), 0.f);
        xorDataSource->addTrainingData(columnVector({0.f, 1.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 0.f}), 1.f);
        xorDataSource->addTrainingData(columnVector({1.f, 1.f}), 0.f);

        cout << "Test with sigmoid" << endl;
        auto neuralNetwork = neuralNetworkBuilder(OptimizerType::sgd)
                ->addInputLayer(xorDataSource->getGivenShape(), 5, LayerType::full, ActivationType::tanhDefault)
                ->addOutputLayer(xorDataSource->getExpectedShape(), ActivationType::sigmoid)->setUseBias(true)
                ->build();

        // good enough results if you round to 0 or 1:
        // 32-bit input node: input size 4,    1000 epochs (8 + 4x4x2  + 8 = 48 bytes)
        // 16-bit input node: input size 4,    1000 epochs (8 + 2x4x2  + 8 = 32 bytes)
        //  8-bit input node: input size 32!!, 1500 epochs (8 + 1x32x2 + 8 = 80 bytes)
        // Clearly, 8-bit doesn't work well for memory savings or quality results in this case.
        neuralNetwork->useHighPrecisionExitStrategy();
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