//
// Created by Erik Hyrkas on 11/28/2022.
// Copyright 2022. Usable under MIT license.
//
#include <memory>
#include "../ml/model.hpp"
#include "../training_data/data_decoder.hpp"

using namespace std;
using namespace happyml;
using namespace happymldsl;

int main() {
    try {
        vector<string> categoryLabels{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        auto expectedEncoder = make_shared<TextToUniqueCategoryEncoder>(categoryLabels);
        auto givenEncoder = make_shared<TextToPixelEncoder>();
        // making the shape square (28x28) just to test the auto-flattening capabilities of the network.
        //"..\\data\\mnist_test.csv"
        cout << "Loading training data..." << endl;
        auto mnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_train.csv", ',',
                                                                                   true, false, true,
                                                                                   1, 28 * 28,
                                                                                   vector<size_t>{1, 10, 1},
                                                                                   vector<size_t>{28, 28, 1},
                                                                                   expectedEncoder, givenEncoder);
        cout << "Loaded training data." << endl;
        cout << "Loading test data..." << endl;
        auto testMnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_test.csv", ',',
                                                                                       true, false, true,
                                                                                       1, 28 * 28,
                                                                                       vector<size_t>{1, 10, 1},
                                                                                       vector<size_t>{28, 28, 1},
                                                                                       expectedEncoder, givenEncoder);
        cout << "Loaded test data." << endl;
        auto neuralNetwork = neuralNetworkBuilder()
                ->setModelName("mnist_conv2d_example")
                ->setModelRepo("../repo/")
                ->addInput(mnistDataSource->getGivenShape(), 1, 3, convolution2dValid,
                           ActivationType::relu)->setUseBias(false)
                ->addNode(100, full, ActivationType::relu)->setUseBias(false)
                ->addOutput(mnistDataSource->getExpectedShape(), sigmoidApprox)
                ->build();
//        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(mnistDataSource, testMnistDataSource, 4);
        // Trained 20 epochs using a batch size of 4 in 52 minutes with a loss of 0.009784.

        cout << fixed << setprecision(2);
        testMnistDataSource->restart();
        auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
        size_t limit = 50;
        auto nextRecord = testMnistDataSource->nextRecord();
        while (nextRecord && limit > 0) {
            auto prediction = decoder->decode(neuralNetwork->predictOne(nextRecord->getFirstGiven()));
            cout << "mnist truth: " << decoder->decode(nextRecord->getFirstExpected()) << " happyml prediction: "
                 << prediction
                 << endl;
            nextRecord = testMnistDataSource->nextRecord();
            limit--;
        }
        cout << fixed << setprecision(4) << "Loss: " << loss;


        // testing save logic:
//        neuralNetwork->saveWithOverwrite();
//        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("mnist_conv2d_example",
//                                                                "../repo/");
//        float testLoss = loadedNeuralNetwork->test(testMnistDataSource);
//        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}