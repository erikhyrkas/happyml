//
// Created by ehyrk on 11/24/2022.
//

#ifndef MICROML_SGD_OPTIMIZER_HPP
#define MICROML_SGD_OPTIMIZER_HPP

#include "neural_network.hpp"
#include "optimizer.hpp"

using namespace std;

// stochastic gradient descent (SGD) is a trivial form of gradient decent that works well at finding generalized results.
// It isn't as popular as Adam, when it comes to optimizers, since it is slow at finding an optimal answer, but
// I've read that it is better at "generalization", which is finding a solution that works for many inputs.
//
// I'm only including it in microml as a starting point to prove everything works, and since it is so simple compared to
// Adam, it lets me test the rest of the code with less fear that I've made a mistake in the optimizer itself.
//
// If you wanted to visualize a tensor, you might think of it as a force pushing in a direction.
// A gradient is a type of tensor pointing toward the fastest improvement.
// Weights are values we use to show how important or unimportant an input is. A neural network has many steps, many of which
// have weights that we need to optimize.
// When we say "optimize", we mean: find the best weights to allow us to make predictions given new input data.
// Stochastic means random.
// So, Stochastic Gradient Descent is using training data in a random order to find the best set of weights to make predictions (inferences)
// given future input data.
namespace microml {
    struct SGDLearningState {
        float learning_rate;
    };

    class SGDFullyConnectedNeurons : public NeuralNetworkFunction {
    public:
        SGDFullyConnectedNeurons(size_t input_size, size_t output_size, bool use_32_bit, const shared_ptr<SGDLearningState> &learning_state) {
            this->input_shapes = vector<vector<size_t>>{{1, input_size, 1}};
            this->output_shapes = vector<vector<size_t>>{{1, output_size, 1}};
            this->weights = make_shared<TensorFromRandom>(input_size, output_size, 1, 14);
            this->use_32_bit = use_32_bit;
            this->learning_state = learning_state;
        }

        vector<vector<size_t>> getInputShapes() {
            return input_shapes;
        }

        vector<vector<size_t>> getOutputShapes() {
            return output_shapes;
        }

        // predicting
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) override {
            last_input = input[0];
            return make_shared<TensorDotTensorView>(last_input, weights);
        }

        // learning
        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            // find the error
            auto weights_transposed = make_shared<TensorTransposeView>(weights);
            auto input_error = make_shared<TensorDotTensorView>(output_error, weights_transposed);

            // update weights
            auto input_transposed = make_shared<TensorTransposeView>(last_input);
            auto weights_error = make_shared<TensorDotTensorView>(input_transposed, output_error);
            auto weights_error_at_learning_rate = make_shared<TensorMultiplyByScalarView>(weights_error,
                                                                                          learning_state->learning_rate);
            auto adjusted_weights = make_shared<TensorMinusTensorView>(weights, weights_error_at_learning_rate);
            if (use_32_bit) {
                weights = make_shared<FullTensor>(*adjusted_weights);
            } else {
                weights = make_shared<QuarterTensor>(*adjusted_weights, 14, 0);
            }

            last_input.reset();
            return shared_ptr<BaseTensor>{input_error};
        }

    private:
        shared_ptr<BaseTensor> weights;
        shared_ptr<BaseTensor> last_input;
        bool use_32_bit;
        vector<vector<size_t>> input_shapes;
        vector<vector<size_t>> output_shapes;
        shared_ptr<SGDLearningState> learning_state;
    };

    class SGDBias : public NeuralNetworkFunction {
    public:
        SGDBias(size_t input_size, size_t output_size, bool use_32_bit, const shared_ptr<SGDLearningState> &learning_state) {
            this->input_shapes = vector<vector<size_t>>{{1, input_size, 1}};
            this->output_shapes = vector<vector<size_t>>{{1, output_size, 1}};
            this->bias = make_shared<TensorFromRandom>(1, output_size, 1, 14);
            this->use_32_bit = use_32_bit;
            this->learning_state = learning_state;
        }

        vector<vector<size_t>> getInputShapes() {
            return input_shapes;
        }

        vector<vector<size_t>> getOutputShapes() {
            return output_shapes;
        }

        // predicting
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input) override {
            last_input = input[0];
            return make_shared<TensorAddTensorView>(last_input, bias);
        }

        // learning
        shared_ptr<BaseTensor> backward(const shared_ptr<BaseTensor> &output_error) override {
            auto bias_error_at_learning_rate = std::make_shared<TensorMultiplyByScalarView>(output_error,
                                                                                            learning_state->learning_rate);
            auto adjusted_bias = std::make_shared<TensorMinusTensorView>(bias, bias_error_at_learning_rate);
            if (use_32_bit) {
                bias = std::make_shared<FullTensor>(*adjusted_bias);
            } else {
                bias = std::make_shared<QuarterTensor>(*adjusted_bias, 14, 0);
            }

            last_input.reset();
            return output_error; // TODO: partial derivative of bias would always be 1, so we pass along original error. I'm fairly sure this is right.
        }

    private:
        shared_ptr<BaseTensor> bias;
        shared_ptr<BaseTensor> last_input;
        bool use_32_bit;
        vector<vector<size_t>> input_shapes;
        vector<vector<size_t>> output_shapes;
        shared_ptr<SGDLearningState> learning_state;
    };

    class SGDOptimizer: public Optimizer {
    public:
        explicit SGDOptimizer(float learning_rate) {
            this->sgdLearningState = make_shared<SGDLearningState>();
            this->sgdLearningState->learning_rate = learning_rate;
        }

        shared_ptr<NeuralNetworkFunction> createFullyConnectedNeurons(size_t input_size, size_t output_size, bool use_32_bit) override  {
            return make_shared<SGDFullyConnectedNeurons>(input_size, output_size, use_32_bit, sgdLearningState);
        }

        shared_ptr<NeuralNetworkFunction> createBias(size_t input_size, size_t output_size, bool use_32_bit) override {
            return make_shared<SGDBias>(input_size, output_size, use_32_bit, sgdLearningState);
        }

    private:
        shared_ptr<SGDLearningState> sgdLearningState;
    };
}
#endif //MICROML_SGD_OPTIMIZER_HPP
