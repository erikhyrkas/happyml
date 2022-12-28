//
// Created by Erik Hyrkas on 12/7/2022.
// Copyright 2022. Usable under MIT license.
//
#include <iostream>
#include "../util/unit_test.hpp"
#include "../ml/model.hpp"

using namespace happymldsl;
using namespace happyml;
using namespace std;


void testSimpleConv2DBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(10, 10, 1, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 1, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 1, convolution2dValid, tanhApprox)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    ASSERT_TRUE(loss < 0.01);

}

void testSimpleConv2DNoBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(10, 10, 1, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 1, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 1, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    ASSERT_TRUE(loss < 0.01);

}


void testConv2DWithFilterNoBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(4, 4, 1, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->addNode(1, 3, convolution2dValid, tanhApprox)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    cout << "Loss: " << loss << endl;
    ASSERT_TRUE(loss < 0.1);
}

void testConv2DWithFilterBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(4, 4, 1, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->addNode(1, 3, convolution2dValid, tanhApprox)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, convolution2dValid, tanhApprox)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    cout << "Loss: " << loss << endl;
    ASSERT_TRUE(loss < 0.1);
}

void testConv2DComplexNoBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(4, 4, 2, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()->setLearningRate(0.01f)
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, convolution2dValid, relu)->setUseBias(
                    false)
            ->addNode(1, 3, convolution2dValid, relu)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, convolution2dValid, sigmoidApprox)->setUseBias(
                    false)
            ->build();
    // it takes 500,000 epochs to get the results fairly close, which takes awhile,
    // so I'll just demonstrate that it does it close enough. If you want it to go faster, use bias.
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    cout << "Loss: " << loss << endl;
    ASSERT_TRUE(loss < 0.1);
}


void testConv2DComplexBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(4, 4, 2, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, convolution2dValid, relu)->setUseBias(
                    false)
            ->addNode(1, 3, convolution2dValid, relu)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, convolution2dValid, sigmoidApprox)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);

    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    cout << "Loss: " << loss << endl;
    ASSERT_TRUE(loss < 0.1);
}


void testConv2DComplexTanhBias() {
    auto conv2dDataSource = make_shared<InMemoryTrainingDataSet>();
    // given input, expected result
    conv2dDataSource->addTrainingData(randomTensor(10, 10, 1, 0.f, 1.f), randomTensor(4, 4, 2, 0.f, 1.f));

    auto neuralNetwork = neuralNetworkBuilder()->setLearningRate(0.01)
            ->addInput(conv2dDataSource->getGivenShape(), 1, 3, convolution2dValid, tanhApprox)->setUseBias(
                    false)
            ->addNode(1, 3, convolution2dValid, tanhApprox)->setUseBias(false)
            ->addOutput(conv2dDataSource->getExpectedShape(), 3, convolution2dValid, tanhApprox)
            ->build();
    float loss = neuralNetwork->train(conv2dDataSource);
    conv2dDataSource->restart();
    auto record = conv2dDataSource->nextRecord();
    auto result = neuralNetwork->predict(record->getFirstGiven());
    cout << "Result: " << endl;
    result[0]->print();
    cout << "Expected: " << endl;
    record->getFirstExpected()->print();
    cout << "Loss: " << loss << endl;
    ASSERT_TRUE(loss < 0.1);
}

int main() {
    try {
        testSimpleConv2DNoBias();
        testSimpleConv2DBias();
        testConv2DWithFilterNoBias();
        testConv2DWithFilterBias();
        testConv2DComplexNoBias();
        testConv2DComplexBias();
        testConv2DComplexTanhBias();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}