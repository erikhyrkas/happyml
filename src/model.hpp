//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_MODEL_HPP
#define MICROML_MODEL_HPP
#include <iostream>
#include "data_source.hpp"
#include "optimizer.hpp"
#include "sgd.hpp"

using namespace microml;

namespace micromldsl {

    enum ModelType { sgd };
    enum LossType { mse };

class MicromlDSL : public enable_shared_from_this<MicromlDSL> {
public:
    explicit MicromlDSL(ModelType modelType) {
        this->modelType = modelType;
        this->learning_rate = 0.1;
    }

    shared_ptr<MicromlDSL> setLearningRate(float learningRate) {
        this->learning_rate = learningRate;
        return shared_from_this();
    }

    shared_ptr<MicromlDSL> setLossFunction(LossType lossType) {
        this->lossType = lossType;
        return shared_from_this();
    }

    shared_ptr<NeuralNetworkForTraining> build() {
        shared_ptr<LossFunction> lossFunction;
        switch(lossType) {
            case LossType::mse:
                lossFunction = make_shared<MeanSquaredErrorLossFunction>();
                break;
            default:
                lossFunction = make_shared<MeanSquaredErrorLossFunction>();
        }
        shared_ptr<Optimizer> optimizer;
        switch(modelType) {
            case ModelType::sgd:
                optimizer = make_shared<SGDOptimizer>(learning_rate);
                break;
            default:
                optimizer = make_shared<SGDOptimizer>(learning_rate);
        }

        auto neuralNetwork = make_shared<NeuralNetworkForTraining>(lossFunction);

        return neuralNetwork;
    }
private:
    ModelType modelType;
    LossType lossType;
    float learning_rate;
};

shared_ptr<MicromlDSL> createSGDModel() {
    auto result = make_shared<MicromlDSL>(sgd);
    return result;
}

}

#endif //MICROML_MODEL_HPP
