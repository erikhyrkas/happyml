//
// Created by ehyrk on 11/24/2022.
//

#include <memory>
#include "model.hpp"
#include "file_reader.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;

int main() {
    try {
        auto lineReader = make_shared<TextLineFileReader>("..\\data\\mnist_test.csv");
        int i = 0;
        while(lineReader->hasNext()) {
            cout << "Record: " << i << endl;
            string record = lineReader->nextLine();
            cout << record << endl;
            cout << "-----" << endl;
            i++;
            if( i > 2) {
                break;
            }
        }

        auto textFileReader = make_shared<DelimitedTextFileReader>("..\\data\\mnist_test.csv", ',');
        i = 0;
        while(textFileReader->hasNext()) {
            cout << "Record: " << i << endl;
            vector<string> record = textFileReader->nextRecord();
            for(auto field : record) {
                cout << field << '|';
            }
            cout << endl << "-----" << endl;
            i++;
            if( i > 2) {
                break;
            }
        }
        auto xorDataSource = make_shared<TestTrainingDataSet>();
        // given input, expected result
        xorDataSource->addTrainingData(column_vector({0.f, 0.f}), 0.f);
        xorDataSource->addTrainingData(column_vector({0.f, 1.f}), 1.f);
        xorDataSource->addTrainingData(column_vector({1.f, 0.f}), 1.f);
        xorDataSource->addTrainingData(column_vector({1.f, 1.f}), 0.f);

        auto neuralNetwork = neuralNetworkBuilder()
                ->addInput(xorDataSource->getGivenShape(), 3, NodeType::full, ActivationType::tanh)
                ->addOutput(xorDataSource->getExpectedShape(), ActivationType::tanh)
                ->build();
        neuralNetwork->train(xorDataSource, 1);

        cout << fixed << setprecision(2);
        cout << "0 xor 0 = 0 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 0.f})) << endl;
        cout << "0 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({0.f, 1.f})) << endl;
        cout << "1 xor 0 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 0.f})) << endl;
        cout << "1 xor 1 = 1 Prediction: " << neuralNetwork->predict_scalar(column_vector({1.f, 1.f})) << endl;

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}