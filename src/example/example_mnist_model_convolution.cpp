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
        // category labels, expected shape, and value:
        // The number 3:
        // [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        // The number 8:
        // [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        // Category labels let us map those arrays back to a number.
        vector<string> categoryLabels{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        auto expectedEncoder = make_shared<TextToUniqueCategoryEncoder>(categoryLabels);
        auto givenEncoder = make_shared<TextToPixelEncoder>();
        // making the shape square (28x28) just to test the auto-flattening capabilities of the network.
        cout << "Loading training data..." << endl;

        auto mnistDataSource = loadDelimitedValuesDataset("..\\data\\mnist_train.csv", ',',
                                                                                   true, false, true,
                                                                                   1, 28 * 28,
                                                                                   vector<size_t>{1, 10, 1},
                                                                                   vector<size_t>{28, 28, 1},
                                                                                   expectedEncoder, givenEncoder);
        cout << "Loaded training data." << endl;
        cout << "Loading test data..." << endl;
        auto testMnistDataSource = loadDelimitedValuesDataset("..\\data\\mnist_test.csv", ',',
                                                                                       true, false, true,
                                                                                       1, 28 * 28,
                                                                                       vector<size_t>{1, 10, 1},
                                                                                       vector<size_t>{28, 28, 1},
                                                                                       expectedEncoder, givenEncoder);

        cout << "Loaded test data." << endl;
        // Here we define our convolutional neural network (CNN):
        // * We give it a model name and repo, so we can save and load this model later.
        // * You'll spot activation functions being used, they adjust the signal between neurons
        // * We tell it the type of input to expect into the first layer
        //   * A convolutional layer applies filters to the image to help us find interesting qualities
        // * We define a fully connected layer as the second layer
        //   * This lets us map those interesting qualities back to our label
        // * We then define a final fully connected layer that has the appropriate output shape
        //   * The sigmoid activation function gives us a probability of a given label
        auto neuralNetwork = neuralNetworkBuilder(happyml::adam)
                ->setModelName("mnist_conv2d_example")
                ->setModelRepo("../repo/")
                ->addInput(mnistDataSource->getGivenShape(), 1, 3, convolution2dValid,
                           ActivationType::relu)->setUseBias(false)
                ->addNode(100, full, ActivationType::relu)->setUseBias(false)
                ->addOutput(mnistDataSource->getExpectedShape(), sigmoidApprox)
                ->build();

//        neuralNetwork->useLowPrecisionExitStrategy();
//        neuralNetwork->useHighPrecisionExitStrategy();

        // batch size impacts the number of images we evaluate at a time.
        float loss = neuralNetwork->train(mnistDataSource, testMnistDataSource, 4);
        // Trained 20 epochs using a batch size of 4 in 52 minutes with a loss of 0.009784.

        cout << fixed << setprecision(2);
        testMnistDataSource->restart();
        auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
        size_t limit = 50;
        auto nextRecord = testMnistDataSource->nextRecord();
        while (nextRecord && limit > 0) {
            // Here we predict one test record at a time
            auto prediction = decoder->decode(neuralNetwork->predictOne(nextRecord->getFirstGiven()));
            // mapping the predicted value (probability array) to a label
            cout << "mnist truth: " << decoder->decode(nextRecord->getFirstExpected()) << " happyml prediction: "
                 << prediction
                 << endl;
            nextRecord = testMnistDataSource->nextRecord();
            limit--;
        }
        cout << fixed << setprecision(4) << "Loss: " << loss;

        // save logic:
//        neuralNetwork->saveWithOverwrite();
//        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("mnist_conv2d_example",
//                                                                "../repo/");
//        float testLoss = loadedNeuralNetwork->test(testMnistDataSource);
//        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}