//
// Created by Erik Hyrkas on 5/8/2023.
// Copyright 2023. Usable under MIT license.
//
#include <memory>
#include "../ml/happyml_dsl.hpp"
#include "../training_data/data_decoder.hpp"
#include "../util/dataset_utils.hpp"
#include "../util/encoder_decoder_builder.hpp"
#include "../lang/execution_context.hpp"

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
        //       using file://../happyml_repo/raw/titanic/train.csv

        // NOTE: If we were doing better data science, we'd split the data set up into a training and test set.
        // For now, we'll use our training set to test. This is not a good practice for real life because it
        // will over fit the model to the training data and not generalize well to new data,
        // but this is fine just to show how to use the library.
        string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
        string result_path = base_path + "titanic/dataset.bin";
        cout << "Loading training data..." << endl;
        auto titanicDataSource = make_shared<BinaryDataSet>(result_path, 0.9);
        auto titanicTestDatasource = make_shared<BinaryDataSet>(result_path, -0.1);

        auto neuralNetwork = neuralNetworkBuilder(OptimizerType::adam)
                ->setLearningRate(0.001f)
                ->setBiasLearningRate(0.001f)
                ->setModelName("titanic_example")
                ->setModelRepo("../happyml_repo/models/")
                ->setLossFunction(LossType::categoricalCrossEntropy)
                ->add_concatenated_input_layer(titanicDataSource->getGivenShapes())
                ->addLayer(32, LayerType::full, ActivationType::relu)->setUseBias(true)->setUseL2Regularization(true)
                ->addDropoutLayer(0.2f)
                ->addLayer(4, LayerType::full, ActivationType::relu)->setUseBias(true)->setUseL2Regularization(true)
                ->addOutputLayer(titanicDataSource->getExpectedShape(), ActivationType::softmax)->setUseBias(true)
                ->build();
        neuralNetwork->setExitStrategy(make_shared<DefaultExitStrategy>(50,
                                                                        NINETY_DAYS_MS,
                                                                        1000000,
                                                                        0.00001f,
                                                                        1e-8,
                                                                        5,
                                                                        0.05f));
        float loss = neuralNetwork->train(titanicDataSource, titanicTestDatasource, 4)->final_test_loss;

        cout << fixed << setprecision(2);
        titanicDataSource->restart();

        BinaryDatasetReader reader(result_path);
//        vector<shared_ptr<RawDecoder>> given_decoders = build_given_decoders(false, reader);
        vector<shared_ptr<RawDecoder >> expected_decoders = build_expected_decoders(false, reader);
        auto first_decoder = expected_decoders[0];

        size_t limit = 5;
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
        neuralNetwork->saveWithOverwrite();
        auto loadedNeuralNetwork = loadNeuralNetworkForTraining("titanic_example",
                                                                "../happyml_repo/models/");

        float testLoss = loadedNeuralNetwork->test(titanicDataSource);
        cout << fixed << setprecision(2) << "Result testLoss: " << testLoss << endl;

        titanicDataSource->restart();
        float accuracy = neuralNetwork->compute_categorical_accuracy(titanicTestDatasource, expected_decoders);
        cout << "Accuracy: " << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}