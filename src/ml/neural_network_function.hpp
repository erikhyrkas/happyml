//
// Created by ehyrk on 11/26/2022.
//

#ifndef MICROML_NEURAL_NETWORK_FUNCTION_HPP
#define MICROML_NEURAL_NETWORK_FUNCTION_HPP

#include "../types/tensor.hpp"
#include "activation.hpp"
#include "../util/basic_profiler.hpp"

namespace microml {
    class NeuralNetworkFunction {
    public:
        // TODO: Can we switch the return types to unique pointers? would it matter?
        virtual shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) = 0;

        // I read an article here that I thought was interesting:
        // https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d
        virtual shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) = 0;
    };

    class NeuralNetworkActivationFunction : public NeuralNetworkFunction {
    public:
        explicit NeuralNetworkActivationFunction(const shared_ptr<ActivationFunction> &activationFunction) {
            this->activationFunction = activationFunction;
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            // todo: throw error on wrong size input?
            PROFILE_BLOCK(profileBlock);
            const auto& lastInput = input[0];
            if(forTraining) {
                lastInputs.push(lastInput);
            }
            return activationFunction->activate(lastInput);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if(lastInputsSize < 1) {
                throw exception("MBGDFullyConnectedNeurons.backward() called without previous inputs.");
            }
            // TODO: it's really inefficient to calculate the derivative of every previous batch input and average it
            //  but doing an average first and then a derivative isn't right.
            //  I think I'm doing the back propagation incorrectly for mini-batch here.
            shared_ptr<BaseTensor> average_activation_derivative= activationFunction->derivative(lastInputs.front());
            lastInputs.pop();
            while(!lastInputs.empty()) {
                auto nextLastInput = activationFunction->derivative(lastInputs.front());
                lastInputs.pop();
                average_activation_derivative = make_shared<TensorAddTensorView>(average_activation_derivative, nextLastInput);
            }
            if(lastInputsSize > 1) {
                average_activation_derivative = materializeTensor(make_shared<TensorMultiplyByScalarView>(average_activation_derivative, 1.f/(float)lastInputsSize));
            }

            //auto activation_derivative = activationFunction->derivative(average_last_inputs);
            // this really threw me for a loop. I thought that this was supposed to be dot product, rather than
            // an element-wise-multiplication.
            auto base_output_error = make_shared<TensorMultiplyTensorView>(average_activation_derivative, output_error);
            return base_output_error;
        }

    private:
        shared_ptr<ActivationFunction> activationFunction;
        queue<shared_ptr<BaseTensor>> lastInputs;
    };

    class NeuralNetworkFlattenFunction : public NeuralNetworkFunction {
    public:
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw exception("Cannot flatten multiple inputs at the same time. Please merge.");
            }
            const auto& nextInput = input[0];
            originalCols = nextInput->columnCount();
            originalRows = nextInput->rowCount();
            if (originalRows == 1) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return nextInput;
            }
            return make_shared<TensorFlattenToRowView>(nextInput);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            if (originalRows == output_error->rowCount() && originalCols == output_error->columnCount()) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return output_error;
            }
            return make_shared<TensorReshapeView>(output_error, originalRows, originalCols);
        }

    private:
        size_t originalRows{};
        size_t originalCols{};
    };
}
#endif //MICROML_NEURAL_NETWORK_FUNCTION_HPP
