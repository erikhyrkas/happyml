//
// Created by Erik Hyrkas on 11/24/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MBGD_OPTIMIZER_HPP
#define HAPPYML_MBGD_OPTIMIZER_HPP

#include "optimizer.hpp"
#include "../util/tensor_utils.hpp"

using namespace std;


// With gradient descent, a single batch is called Stochastic Gradient Descent.
// A batch with all records is called Batch Gradient Descent. And a batch anywhere
// in between is called Mini-Batch Gradient Descent. Mini-Batch is fastest, handles
// large datasets, and most commonly used of this optimization approach.
//
// stochastic gradient descent (SGD) is a trivial form of gradient decent that works well at finding generalized results.
// It isn't as popular as Adam, when it comes to optimizers, since it is slow at finding an optimal answer, but
// I've read that it is better at "generalization", which is finding a solution that works for many inputs.
//
// I'm only including it in happyml as a starting point to prove everything works, and since it is so simple compared to
// Adam, it lets me test the rest of the code with less fear that I've made a mistake in the optimizer itself.
//
// If you wanted to visualize a tensor, you might think of it as a force pushing in a direction.
// A gradient is a type of tensor (or slope) related to the error in the model pointing toward the fastest improvement.
// Weights are values we use to show how important or unimportant an input is. A neural network has many steps, many of which
// have weights that we need to optimize.
// When we say "optimize", we mean: find the best weights to allow us to make predictions given new input data.
// Stochastic means random.
// So, Stochastic Gradient Descent is using training data in a random order to find the best set of weights to make predictions (inferences)
// given future input data.
namespace happyml {


    class MBGDOptimizer : public BaseOptimizer {
    public:
        explicit MBGDOptimizer(float learningRate, float biasLearningRate)
                : BaseOptimizer(learningRate, biasLearningRate) {}


        int registerForWeightChanges() override {
            return 0;
        }

        int registerForBiasChanges() override {
            return 0;
        }

        shared_ptr<BaseTensor> calculateWeightsChange(int registration_id,
                                                      const shared_ptr<BaseTensor> &weights,
                                                      const shared_ptr<BaseTensor> &weightChanges,
                                                      float mixedPrecisionScale) override {

            const auto adjustedWeightChanges = make_shared<TensorMultiplyByScalarView>(weightChanges,
                                                                                       learningRate *
                                                                                       mixedPrecisionScale);
            const auto adjustedWeights = make_shared<TensorMinusTensorView>(weights,
                                                                            adjustedWeightChanges);
            return adjustedWeights;
        }

        shared_ptr<BaseTensor> calculateBiasChange(int registration_id,
                                                   const shared_ptr<BaseTensor> &bias,
                                                   const shared_ptr<BaseTensor> &loss_gradient,
                                                   float mixedPrecisionScale,
                                                   float current_batch_size) override {
            auto bias_error_at_learning_rate = make_shared<TensorMultiplyByScalarView>(loss_gradient,
                                                                                       biasLearningRate *
                                                                                       mixedPrecisionScale /
                                                                                       current_batch_size);
            auto adjusted_bias = make_shared<TensorMinusTensorView>(bias, bias_error_at_learning_rate);
            return adjusted_bias;
        }

    };
}
#endif //HAPPYML_MBGD_OPTIMIZER_HPP
