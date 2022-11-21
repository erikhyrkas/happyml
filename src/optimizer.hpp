//
// Created by Erik Hyrkas on 11/5/2022.
//

#ifndef MICROML_OPTIMIZER_HPP
#define MICROML_OPTIMIZER_HPP

#include "loss.hpp"
#include "data.hpp"

// Optimizers are the strategy applied to find the optimal results
// The optimizer takes in:
// 1. The results that the model predicted
// 2. The results that the model should have gotten (the "truth")
// The optimizer uses a loss function (which is simply a bit of math to calculate how close a prediction is
// to the true answer) to compare the two and then updates the weights.
class Optimizer {
public:
    virtual void minimize(LossFunction &lossFunction, std::vector<BaseTensor> &predicted, std::vector<BaseTensor> &truth, BaseAssignableTensor &weights) = 0;
};
#endif //MICROML_OPTIMIZER_HPP
