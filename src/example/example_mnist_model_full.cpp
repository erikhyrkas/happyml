//
// Created by Erik Hyrkas on 11/28/2022.
// Copyright 2022. Usable under MIT license.
//
#include <memory>
#include "../ml/happyml_dsl.hpp"
#include "../training_data/data_decoder.hpp"
#include "../util/dataset_utils.hpp"

using namespace std;
using namespace happyml;

int main() {
    try {
        vector<string> categoryLabels{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        auto expectedEncoder = make_shared<TextToUniqueCategoryEncoder>(categoryLabels);
        auto givenEncoder = make_shared<TextToPixelEncoder>();
        // making the shape square (28x28) just to test the auto-flattening capabilities of the network.
        //"..\\happyml_data\\mnist_test.csv"
        //"..\\happyml_data\\mnist_train.csv"
        cout << "Loading training data..." << endl;
        auto mnistDataSource = loadDelimitedValuesDataset("..\\happyml_data\\mnist_train.csv", ',',
                                                          true, false, true,
                                                          1, 28 * 28,
                                                          vector<size_t>{1, 10, 1},
                                                          vector<size_t>{28, 28, 1},
                                                          expectedEncoder, givenEncoder);
        cout << "Loaded training data." << endl;
        cout << "Loading test data..." << endl;
        auto testMnistDataSource = loadDelimitedValuesDataset("..\\happyml_data\\mnist_test.csv", ',',
                                                              true, false, true,
                                                              1, 28 * 28,
                                                              vector<size_t>{1, 10, 1},
                                                              vector<size_t>{28, 28, 1},
                                                              expectedEncoder, givenEncoder);
        cout << "Loaded test data." << endl;
        auto neuralNetwork = neuralNetworkBuilder()
                ->addInputLayer(mnistDataSource->getGivenShape(), 100, LayerType::full, ActivationType::relu)
                ->addLayer(50, LayerType::full, ActivationType::relu)
                ->addOutputLayer(mnistDataSource->getExpectedShape(), ActivationType::sigmoidApprox)->setUseBias(true)
                ->build();
        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(mnistDataSource, testMnistDataSource, 32);
        cout << fixed << setprecision(2);

        testMnistDataSource->restart();
        auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
        size_t limit = 50;
        auto nextRecord = testMnistDataSource->nextRecord();
        while (nextRecord && limit > 0) {
            auto prediction = decoder->decodeBest(neuralNetwork->predictOne(nextRecord->getGiven()[0]));
            cout << "mnist truth: " << decoder->decodeBest(nextRecord->getExpected()[0]) << " happyml prediction: "
                 << prediction
                 << endl;
            nextRecord = testMnistDataSource->nextRecord();
            limit--;
        }
        cout << fixed << setprecision(4) << "Loss: " << loss;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}