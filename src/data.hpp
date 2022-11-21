//
// Created by Erik Hyrkas on 11/5/2022.
//

#ifndef MICROML_DATA_HPP
#define MICROML_DATA_HPP

#include "tensor.hpp"

// Training data has at least two parts:
// 1. The input that you are giving the model
// 2. The expected predictions you are expecting the model to make
//
// Many models only make a single prediction, but there are plenty
// of models that make multiple predictions.
class TrainingPair {
public:
    TrainingPair(std::shared_ptr<BaseTensor> given, std::vector<std::shared_ptr<BaseTensor>> &expected) {
        this->given = given;
        for(std::shared_ptr<BaseTensor> tensor : expected) {
            this->expected.push_back(tensor);
        }
    }
    std::shared_ptr<BaseTensor> getGiven() {
        return given;
    }
    std::shared_ptr<BaseTensor> getExpected(size_t offset) {
        return expected.at(offset);
    }
    size_t getExpectedSize() {
        return expected.size();
    }
private:
    std::shared_ptr<BaseTensor> given;
    std::vector<std::shared_ptr<BaseTensor>> expected;
};

#endif //MICROML_DATA_HPP
