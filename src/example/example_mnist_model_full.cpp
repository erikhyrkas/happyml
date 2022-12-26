//
// Created by Erik Hyrkas on 11/28/2022.
//
#include <memory>
#include "../ml/model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;

int main() {
    try {
        map<string, size_t> categories;
        categories["0"] = 0;
        categories["1"] = 1;
        categories["2"] = 2;
        categories["3"] = 3;
        categories["4"] = 4;
        categories["5"] = 5;
        categories["6"] = 6;
        categories["7"] = 7;
        categories["8"] = 8;
        categories["9"] = 9;
        auto expectedEncoder = make_shared<TextToCategoryEncoder>(categories);
        auto givenEncoder = make_shared<TextToPixelEncoder>();
        // making the shape square (28x28) just to test the auto-flattening capabilities of the network.
        //"..\\data\\mnist_test.csv"
        //"..\\data\\mnist_train.csv"
        cout << "Loading training data..." << endl;
        auto mnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_train.csv", ',',
                                                                                   true, false, true,
                                                                                   1, 28*28,
                                                                                   vector<size_t>{1,10,1},vector<size_t>{28, 28,1},
                                                                                   expectedEncoder, givenEncoder);
        cout << "Loaded training data." << endl;
        cout << "Loading test data..." << endl;
        auto testMnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_test.csv", ',',
                                                                                       true, false, true,
                                                                                       1, 28*28,
                                                                                       vector<size_t>{1,10,1},vector<size_t>{28, 28,1},//vector<size_t>{28,28,1},,vector<size_t>{28,28,1},
                                                                                       expectedEncoder, givenEncoder);
        cout << "Loaded test data." << endl;

        auto neuralNetwork = neuralNetworkBuilder()
                ->addInput(mnistDataSource->getGivenShape(), 100, NodeType::full, relu)->setUseBias(false)
                ->addNode(50, NodeType::full, relu)->setUseBias(false)
                ->addOutput(mnistDataSource->getExpectedShape(), ActivationType::sigmoidApprox)
                ->build();
        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(mnistDataSource, testMnistDataSource, 4);
        // Trained 20 epochs using a batch size of 1 in 269 minutes with a loss of 0.000004.
        // Trained 20 epochs using a batch size of 4 in 92 minutes with a loss of 0.011218.
        // Trained 20 epochs using a batch size of 8 in 67 minutes with a loss of 0.017087.
        // Trained 20 epochs using a batch size of 16 in 54 minutes with a loss of 0.024266.
        // Trained 20 epochs using a batch size of 256 in 42 minutes with a loss of 0.094601.
        cout << fixed << setprecision(2);
        testMnistDataSource->restart();
        size_t limit = 50;
        auto nextRecord = testMnistDataSource->nextRecord();
        while(nextRecord && limit > 0) {
            auto prediction = maxIndex(neuralNetwork->predictOne(nextRecord->getFirstGiven()));
            cout << "mnist truth: " << maxIndex(nextRecord->getFirstExpected()) << " microml prediction: " << prediction << endl;
            nextRecord = testMnistDataSource->nextRecord();
            limit--;
        }
        cout << fixed << setprecision(4) << "Loss: " << loss;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}