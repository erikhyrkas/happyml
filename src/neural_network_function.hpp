//
// Created by ehyrk on 11/26/2022.
//

#ifndef MICROML_NEURAL_NETWORK_FUNCTION_HPP
#define MICROML_NEURAL_NETWORK_FUNCTION_HPP

#include "tensor.hpp"
#include "activation.hpp"

namespace microml {
    class NeuralNetworkFunction {
    public:
        virtual shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) = 0;

        virtual shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) = 0;
    };

    class NeuralNetworkActivationFunction : public NeuralNetworkFunction {
    public:
        explicit NeuralNetworkActivationFunction(const shared_ptr<ActivationFunction> &activationFunction) {
            this->activationFunction = activationFunction;
        }
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) override {
            // todo: throw error on wrong size input?
            last_input = input[0];
            return activationFunction->activate(last_input);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            auto activation_derivative = activationFunction->derivative(last_input);
            // this really threw me for a loop. I thought that this was supposed to be dot product, rather than
            // an element-wise-multiplication.
            auto base_output_error = std::make_shared<TensorMultiplyTensorView>(activation_derivative, output_error);
            last_input = nullptr;
            return base_output_error;
        }

    private:
        shared_ptr<ActivationFunction> activationFunction;
        shared_ptr<BaseTensor> last_input;
    };
}
#endif //MICROML_NEURAL_NETWORK_FUNCTION_HPP
