//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TRAINING_PAIR_HPP
#define HAPPYML_TRAINING_PAIR_HPP

#include "../types/base_tensors.hpp"
#include "../util/tensor_utils.hpp"

using namespace std;

namespace happyml {

// Training data has at least two parts:
// 1. The input that you are giving the model
// 2. The expected predictions you are expecting the model to make
//
// Many models only make a single prediction, but there are plenty
// of models that make multiple predictions.
    class TrainingPair {
    public:
//        TrainingPair(vector<shared_ptr<BaseTensor>> &given, vector<shared_ptr<BaseTensor>> &expected) {
//            for (const shared_ptr<BaseTensor> &tensor: given) {
//                this->given.push_back(tensor);
//            }
//            for (const shared_ptr<BaseTensor> &tensor: expected) {
//                this->expected.push_back(tensor);
//            }
//        }

        TrainingPair(const vector<shared_ptr<BaseTensor>> &given, const vector<shared_ptr<BaseTensor>> &expected) {
            for (const shared_ptr<BaseTensor> &tensor: given) {
                this->given.push_back(tensor);
            }
            for (const shared_ptr<BaseTensor> &tensor: expected) {
                this->expected.push_back(tensor);
            }
        }

        TrainingPair(const shared_ptr<BaseTensor> &given, const shared_ptr<BaseTensor> &expected) {
            this->given.push_back(given);
            this->expected.push_back(expected);
        }

        TrainingPair(const vector<float> &given, const vector<float> &expected) {
            this->given.push_back(columnVector(given));
            this->expected.push_back(columnVector(expected));
        }

        vector<shared_ptr<BaseTensor>> getGiven() {
            return given;
        }

        size_t getGivenSize() {
            return given.size();
        }

        vector<shared_ptr<BaseTensor>> getExpected() {
            return expected;
        }

        size_t getExpectedSize() {
            return expected.size();
        }

    private:
        vector<shared_ptr<BaseTensor>> given;
        vector<shared_ptr<BaseTensor>> expected;
    };
}
#endif //HAPPYML_TRAINING_PAIR_HPP
