//
// Created by Erik Hyrkas on 5/8/2023.
// Copyright 2023. Usable under MIT license.
//
#include <memory>
#include "../ml/happyml_dsl.hpp"
#include "../training_data/data_decoder.hpp"
#include "../util/dataset_utils.hpp"
#include "../util/happyml_paths.hpp"
#include "../util/encoder_decoder_builder.hpp"

using namespace std;
using namespace happyml;

string to_survived(string &val) {
    if (val == "1") {
        return "survived";
    }
    return "died    ";
}

int main() {
    try {
        // PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        //           0,       1,     2,   3,  4,  5,    6,    7,     8,   9,   10,      11
        //
        // Valid Date Types can be: image, label, number, text
        //
        // create dataset titanic
        //       with header
        //       with expected label at 1     # Survived
        //       with given label    at 2     # Pclass
        //       with given label    at 4     # Sex
        //       with given number   at 5     # Age
        //       with given number   at 6     # SibSp
        //       with given number   at 7     # Parch
        //       with given number   at 9     # Fare
        //       with given label    at 11    # Embarked
        //       using file://../data/titanic/train.csv
        string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
        string result_path = base_path + "titanic/dataset.bin";
        cout << "Loading training data..." << endl;
        happyml::BinaryDatasetReader reader(result_path);
        auto titanicDataSource = make_shared<BinaryDataSet>(result_path);

        auto neuralNetwork = neuralNetworkBuilder()
                ->add_concatenated_input_layer(titanicDataSource->getGivenShapes())
                ->addLayer(64, LayerType::full, ActivationType::relu)->setUseBias(false)
                ->addLayer(64, LayerType::full, ActivationType::relu)->setUseBias(false)
                ->addLayer(8, LayerType::full, ActivationType::relu)->setUseBias(false)
                ->addOutputLayer(titanicDataSource->getExpectedShape(), ActivationType::sigmoidApprox)
                ->build();
        neuralNetwork->useHighPrecisionExitStrategy();
        float loss = neuralNetwork->train(titanicDataSource, 1);

        cout << fixed << setprecision(2);
        titanicDataSource->restart();
//        vector<shared_ptr<happyml::RawDecoder>> given_decoders = build_given_decoders(false, reader);
        vector<shared_ptr<happyml::RawDecoder >> expected_decoders = build_expected_decoders(false, reader);
        auto first_decoder = expected_decoders[0];

        size_t limit = 50;
        auto nextRecord = titanicDataSource->nextRecord();
        while (nextRecord && limit > 0) {
            auto prediction = first_decoder->decodeBest(neuralNetwork->predictOne(nextRecord->getGiven()));
            auto truth = first_decoder->decodeBest(nextRecord->getExpected()[0]);
            cout << "titanic truth: " << to_survived(truth) << " -> happyml prediction: "
                 << to_survived(prediction)
                 << endl;
            nextRecord = titanicDataSource->nextRecord();
            limit--;
        }
        cout << fixed << setprecision(4) << "Loss: " << loss;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}