//
// Created by Erik Hyrkas on 12/7/2022.
//
#include <iostream>
#include "../util/unit_test.hpp"
#include "../ml/model.hpp"
#include "../util/tensor_utils.hpp"

using namespace micromldsl;
using namespace microml;
using namespace std;


void testSimpleConv2D() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1), randomTensor(10, 10, 1));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 1, micromldsl::convolution2d, tanh_approx)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 1, micromldsl::convolution2d, tanh_approx)
            ->build();
    neuralNetwork->train(conv2dDataSource, 30000, 1);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
}
void testConv2D() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1), randomTensor(4, 4, 2));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, micromldsl::convolution2d, tanh_approx)->setUseBias(false)
            ->addNode(1, 3, micromldsl::convolution2d, tanh_approx)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, micromldsl::convolution2d, tanh_approx)->setUseBias(false)
            ->build();
    neuralNetwork->train(conv2dDataSource, 30, 1);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
}

int main() {
    try {
        testSimpleConv2D();
//        testConv2D();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}