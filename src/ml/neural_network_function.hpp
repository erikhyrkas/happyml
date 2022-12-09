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
        virtual shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) = 0;

        // I read an article here that I thought was interesting:
        // https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d
        virtual shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) = 0;
    };

    class NeuralNetworkActivationFunction : public NeuralNetworkFunction {
    public:
        explicit NeuralNetworkActivationFunction(const shared_ptr<ActivationFunction> &activationFunction) {
            this->activationFunction = activationFunction;
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) override {
            // todo: throw error on wrong size input?
            PROFILE_BLOCK(profileBlock);
            last_input = input[0];
            return activationFunction->activate(last_input);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
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

    class NeuralNetworkFlattenFunction : public NeuralNetworkFunction {
    public:
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw exception("Cannot flatten multiple inputs at the same time. Please merge.");
            }
            const auto& last_input = input[0];
            original_cols = last_input->column_count();
            original_rows = last_input->row_count();
            if (original_rows == 1) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return last_input;
            }
            return make_shared<TensorFlattenToRowView>(last_input);
        }

        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            if (original_rows == output_error->row_count() && original_cols == output_error->column_count()) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return output_error;
            }
            return make_shared<TensorReshapeView>(output_error, original_rows, original_cols);
        }

    private:
        size_t original_rows{};
        size_t original_cols{};
    };
}
#endif //MICROML_NEURAL_NETWORK_FUNCTION_HPP
