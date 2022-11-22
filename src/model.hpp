//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_MODEL_HPP
#define MICROML_MODEL_HPP
#include <iostream>
#include "data_source.hpp"
#include "optimizer.hpp"
#include "layer.hpp"

// A model is a combination of configuration and code
// There are two core purposes of a model:
// 1. to learn from inputs
// 2. to make a prediction (an inference) from inputs based on prior learning
// The configuration
class MicromlModel {
public:
    MicromlModel() {

    }
    MicromlModel(std::string config_filename) {
        load_config(config_filename);
    }

    void load_config(std::string config_filename) {

    }

    void save_config(std::string config_filename) {

    }

    void train(BaseMicromlDataSource &dataSource) {
        EmptyDataSource validationDataSource;
        train(dataSource, validationDataSource);
    }

    void train(BaseMicromlDataSource &dataSource, float percent_for_validation) {
        if(percent_for_validation < 0 || percent_for_validation >= 1.0) {
            throw std::exception("Validation split has to be 0 or more and less than 1. 0.2 is a good default.");
        }
        if(percent_for_validation > 0) {
            size_t validation_records = dataSource.record_count()*percent_for_validation;

        } else {
            EmptyDataSource validationDataSource;
            train(dataSource, validationDataSource);
        }
    }
    void train(BaseMicromlDataSource &trainingDataSource, BaseMicromlDataSource &validationDataSource) {
        // the basic flow is that we make a prediction and then send feedback
        // to the model to adjust it to be slightly closer to right.
        // We have to make small adjustments because we want a generalized solution
        // and while a specific set of weights might give us 100% accuracy in a
        // specific case, we want the highest accuracy we can get for all cases.
        // We have two data sources: One for training and one for validation.
        // It's important that the validation set does not overlap with the training set.

        // we do training in epocs and steps. Each step is a single batch from the training data set
        // followed by a single batch from the validation data set.


    }

    void infer(BaseTensor input) {

    }

private:
    std::string working_folder;
    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Neuron> head_layer;

};
#endif //MICROML_MODEL_HPP
