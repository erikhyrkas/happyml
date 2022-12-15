//
// Created by Erik Hyrkas on 11/28/2022.
//
#include <memory>
#include "../ml/model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;



void testMnistFull() {
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
    //"..\\test_data\\small_mnist_format.csv"
    //"..\\data\\mnist_test.csv"
    //"..\\data\\mnist_train.csv"
    auto mnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_train.csv", ',',
                                                                             true, false, true,
                                                                             1, 28*28,
                                                                             vector<size_t>{1,10,1},vector<size_t>{28, 28,1},
                                                                             expectedEncoder, givenEncoder);
    cout << "Loaded training data." << endl;

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(mnistDataSource->getGivenShape(), 100, NodeType::full, ActivationType::relu)->setUseBias(false)
            ->addNode(50, NodeType::full, ActivationType::relu)->setUseBias(false)
            ->addOutput(mnistDataSource->getExpectedShape(), ActivationType::sigmoid_approx)
            ->build();
    neuralNetwork->train(mnistDataSource, 20, 1);
    // using a batch size of 1:
    //    2 ms to predict     Epoch:     20/20        Batch: 60000/60000 Batch Size:   1      Loss:    0.000004
    //   11 ms to learn       Epoch:     20/20        Batch: 60000/60000 Batch Size:   1      Loss:    0.000004
    //
    //Finished training in 269 minutes.
    //Loaded test data.
    //mnist truth: 7 microml prediction: 7
    //mnist truth: 2 microml prediction: 2
    //mnist truth: 1 microml prediction: 1
    //mnist truth: 0 microml prediction: 0
    //mnist truth: 4 microml prediction: 4
    //mnist truth: 1 microml prediction: 1
    //mnist truth: 4 microml prediction: 4
    auto testMnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_test.csv", ',',
                                                                               true, false, true,
                                                                               1, 28*28,
                                                                               vector<size_t>{1,10,1},vector<size_t>{28, 28,1},//vector<size_t>{28,28,1},,vector<size_t>{28,28,1},
                                                                               expectedEncoder, givenEncoder);
    cout << "Loaded test data." << endl;
    cout << fixed << setprecision(2);
    auto nextRecord = testMnistDataSource->nextRecord();
    while(nextRecord) {
        auto prediction = maxIndex(neuralNetwork->predictOne(nextRecord->getFirstGiven()));
        cout << "mnist truth: " << maxIndex(nextRecord->getFirstExpected()) << " microml prediction: " << prediction << endl;
        nextRecord = testMnistDataSource->nextRecord();
    }
}

void testMnistConvolution() {
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
//"..\\test_data\\small_mnist_format.csv"
//"..\\data\\mnist_test.csv"
    auto mnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\data\\mnist_test.csv", ',',
                                                                               true, false, true,
                                                                               1, 28*28,
                                                                               vector<size_t>{1,10,1},vector<size_t>{28, 28,1},//vector<size_t>{28,28,1},
                                                                               expectedEncoder, givenEncoder);
    cout << "Loaded training data." << endl;

    auto neuralNetwork = neuralNetworkBuilder()
            ->addInput(mnistDataSource->getGivenShape(), 100, full, tanh_approx)
            ->addNode(50, full, tanh_approx)
            ->addOutput(mnistDataSource->getExpectedShape(), tanh_approx)
            ->build();
    neuralNetwork->train(mnistDataSource, 100, 128);


    auto testMnistDataSource = make_shared<InMemoryDelimitedValuesTrainingDataSet>("..\\test_data\\small_mnist_format.csv", ',',
                                                                                   true, false, true,
                                                                                   1, 28*28,
                                                                                   vector<size_t>{1,10,1},vector<size_t>{28, 28,1},//vector<size_t>{28,28,1},,vector<size_t>{28,28,1},
                                                                                   expectedEncoder, givenEncoder);
    cout << "Loaded test data." << endl;
    cout << fixed << setprecision(2);
    auto nextRecord = testMnistDataSource->nextRecord();
    while(nextRecord) {
        auto prediction = maxIndex(neuralNetwork->predictOne(nextRecord->getFirstGiven()));
        cout << "mnist truth: " << maxIndex(nextRecord->getFirstExpected()) << " microml prediction: " << prediction << endl;
        nextRecord = testMnistDataSource->nextRecord();
    }
}

int main() {
    try {
        testMnistFull();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}