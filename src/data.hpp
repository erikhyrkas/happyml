//
// Created by Erik Hyrkas on 11/5/2022.
//

#ifndef MICROML_DATA_HPP
#define MICROML_DATA_HPP

#include "tensor.hpp"

using namespace std;

namespace microml {

// Training data has at least two parts:
// 1. The input that you are giving the model
// 2. The expected predictions you are expecting the model to make
//
// Many models only make a single prediction, but there are plenty
// of models that make multiple predictions.
class TrainingPair {
public:
    TrainingPair(vector<shared_ptr<BaseTensor>> &given, vector<shared_ptr<BaseTensor>> &expected) {
        for(const shared_ptr<BaseTensor>& tensor : given) {
            this->given.push_back(tensor);
        }
        for(const shared_ptr<BaseTensor>& tensor : expected) {
            this->expected.push_back(tensor);
        }
    }
    TrainingPair(const vector<shared_ptr<BaseTensor>>& given, const vector<shared_ptr<BaseTensor>>& expected) {
        for(const shared_ptr<BaseTensor>& tensor : given) {
            this->given.push_back(tensor);
        }
        for(const shared_ptr<BaseTensor>& tensor : expected) {
            this->expected.push_back(tensor);
        }
    }
    TrainingPair(const vector<float> &given, const vector<float> expected) {
        this->given.push_back(make_shared<FullTensor>(given));
        this->expected.push_back(make_shared<FullTensor>(expected));
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
#endif //MICROML_DATA_HPP
