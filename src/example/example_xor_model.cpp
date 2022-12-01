//
// Created by ehyrk on 11/24/2022.
//

#include <memory>
#include "../model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;


void using_tanh() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(column_vector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(column_vector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 1.f}), 0.f);

    cout << "Test with tanh" << endl;
    auto neuralNetwork = neuralNetworkBuilder()->setLearningRate(0.01)
            ->addInput(xorDataSource->getGivenShape(), 3, NodeType::full, ActivationType::tanh)->set32Bit(false)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::tanh)//->setUseBias(false)
            ->build();
    // Gives very good results at 8,000 epochs, but somewhere under 1,000 epochs the results
    // are occasionally wrong for 0, 0. Below 300 epochs and it's usually wrong for 0, 0.
    // 1,000 epochs is good enough if you round to the nearest whole number, which you would anyway.
    neuralNetwork->train(xorDataSource, 8000, 1);


    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 1.f})) << endl;
}

void using_sigmoid() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(column_vector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(column_vector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 1.f}), 0.f);

    cout << "Test with sigmoid" << endl;
    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(xorDataSource->getGivenShape(), 3, NodeType::full, ActivationType::tanh)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::sigmoid)
            ->build();

    neuralNetwork->train(xorDataSource, 1000);

    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 1.f})) << endl;
}


void using_relu() {
    auto xorDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    xorDataSource->addTrainingData(column_vector({0.f, 0.f}), 0.f);
    xorDataSource->addTrainingData(column_vector({0.f, 1.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 0.f}), 1.f);
    xorDataSource->addTrainingData(column_vector({1.f, 1.f}), 0.f);

    cout << "Test with relu" << endl;
    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(xorDataSource->getGivenShape(), 8, NodeType::full, ActivationType::relu)
            ->addNode(2, NodeType::full, ActivationType::relu)
            ->addOutput(xorDataSource->getExpectedShape(), ActivationType::sigmoid)
            ->build();

    neuralNetwork->train(xorDataSource, 1000);


    cout << fixed << setprecision(2);
    cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 0.f})) << endl;
    cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 1.f})) << endl;
    cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 0.f})) << endl;
    cout << "1 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 1.f})) << endl;
}

int main() {
    try {
        using_tanh();
//        using_sigmoid();
//        using_relu();

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}