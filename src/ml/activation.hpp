//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_ACTIVATION_HPP
#define HAPPYML_ACTIVATION_HPP

#include <iostream>
#include "../types/tensor_views/value_transform_tensor_view.hpp"


// To me, it feels like activation functions are the heart and soul of modern ml.
// Unfortunately, they can be a little hard to understand without some math background.
// I'll do my best to give you the very, very basics:
// * If you haven't had calculus, a derivative of an equation describes the rate the original equation changed its output.
//   Here's a little tutorial that I hope is useful: https://www.mathsisfun.com/calculus/derivatives-introduction.html
//   It might help you visualize to know that: The derivative of X squared is two times X.
//   Also written as: d/dx X^2 = 2X
// * We use the activation function on the way "forward" while we are predicting/inferring.
// * We use the derivative of the activation function on the way "backward" when we are training to adjust our weights.
// * Weights and bias are the numbers we are adjusting so the model learns. Activation functions are concerned with
//   only the weights.
// * I found this article very helpful when trying to remember the math of each:
//   https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
// * You may also find this useful:
//   https://en.wikipedia.org/wiki/Activation_function
namespace happyml {

    class ActivationFunction {
    public:
        virtual std::shared_ptr<BaseTensor> activate(const std::shared_ptr<BaseTensor> &input) = 0;

        virtual std::shared_ptr<BaseTensor> derivative(const std::shared_ptr<BaseTensor> &input) = 0;
    };

}
#endif //HAPPYML_ACTIVATION_HPP
