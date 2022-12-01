//
// Created by ehyrk on 11/24/2022.
//

#ifndef MICROML_MBGD_OPTIMIZER_HPP
#define MICROML_MBGD_OPTIMIZER_HPP

#include "neural_network.hpp"
#include "optimizer.hpp"

using namespace std;


// With gradient descent, a single batch is called Stocastich Gradient Descent.
// A batch with all records is called Batch Gradient Descent. And a batch anywhere
// in between is called Mini-Batch Gradient Descent. Mini-Batch is fastest, handles
// large datasets, and most commonly used of this optimization approach.
//
// stochastic gradient descent (SGD) is a trivial form of gradient decent that works well at finding generalized results.
// It isn't as popular as Adam, when it comes to optimizers, since it is slow at finding an optimal answer, but
// I've read that it is better at "generalization", which is finding a solution that works for many inputs.
//
// I'm only including it in microml as a starting point to prove everything works, and since it is so simple compared to
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
namespace microml {
    struct MBGDLearningState {
        float learning_rate;
    };

    class MBGDFullyConnectedNeurons : public NeuralNetworkFunction {
    public:
        MBGDFullyConnectedNeurons(size_t input_size, size_t output_size, bool use_32_bit,
                                  const shared_ptr<MBGDLearningState> &learning_state) {
            this->input_shapes = vector<vector<size_t>>{{1, input_size, 1}};
            this->output_shapes = vector<vector<size_t>>{{1, output_size, 1}};
            this->weights = make_shared<TensorFromRandom>(input_size, output_size, 1, -0.5f, 0.5f, 42);
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
            shared_ptr<BaseTensor> input_error = make_shared<TensorDotTensorView>(output_error, weights_transposed);

            // update weights
            auto input_transposed = make_shared<TensorTransposeView>(last_input);
            auto weights_error = make_shared<TensorDotTensorView>(input_transposed, output_error);
            auto weights_error_at_learning_rate = make_shared<TensorMultiplyByScalarView>(weights_error,
                                                                                          learning_state->learning_rate);
            auto adjusted_weights = make_shared<TensorMinusTensorView>(weights, weights_error_at_learning_rate);

//            cout << endl << "Min: " << adjusted_weights->min() << " Max: " <<adjusted_weights->max() <<endl;
//            cout << "output error: " << endl;
//            output_error->print();
//            cout << "weights transposed: " << endl;
//            weights_transposed->print();
//            cout << "input error: " << endl;
//            input_error->print();
//            cout << "last input: " <<endl;
//            last_input->print();
//            cout << "inputs transposed:" << endl;
//            input_transposed->print();
//            cout << "weights: "<<endl;
//            weights->print();
//            cout << "weights error:" <<endl;
//            weights_error->print();
//            cout << "Lr: " << learning_state->learning_rate << endl;
//            cout << "weights error at learning rate:" <<endl;
//            weights_error_at_learning_rate->print();
//            cout << "adjusted weights:" <<endl;
//            adjusted_weights->print();

            if (use_32_bit) {
                float adj_min = adjusted_weights->min();
                float adj_max = adjusted_weights->max();
//                cout << endl << " adj_min: " << adj_min << " adj_max:" << adj_max << endl;

                weights = make_shared<FullTensor>(adjusted_weights);
            } else {
                auto min_max = adjusted_weights->range();
                const float adj_min = min_max.first; // adjusted_weights->min();
                const float adj_max = min_max.second; //adjusted_weights->max();
                int result_bias = 8;
                for(int proposed_bias = 15; proposed_bias >= 8; proposed_bias--) {
                    float bias_max = quarter_to_float(QUARTER_MAX, proposed_bias, 0)*0.8;
                    float bias_min = -bias_max;
                    if(adj_min > bias_min && adj_max < bias_max) {
//                        cout << endl << "proposed_bias: " << proposed_bias << " " << bias_min << " -> " << bias_max << " adj_min: " << adj_min << " adj_max:" << adj_max << endl;
                        result_bias = proposed_bias;
                        break;
                    }
                }
                weights = make_shared<QuarterTensor>(adjusted_weights, result_bias, 0);
//                auto real_weights = make_shared<FullTensor>(adjusted_weights);
//                cout << endl << "8-bit vs 32-bit weights:" << endl;
//                real_weights->print();
//                weights->print();
//                cout << endl;

//                cout << endl << "adjusted:";
//                adjusted_weights->print();
//                cout << endl << "alternate:";
//                auto alternate_weights = make_shared<FullTensor>(*adjusted_weights);
//                alternate_weights->print();
//                cout << endl << "actual:";
//                weights->print();
//                cout << endl << "bias " << result_bias << endl;
//                cout << "assigned weights:" <<endl;
//                weights->print();
            }

            last_input.reset();
            return input_error;
        }

    private:
        shared_ptr<BaseTensor> weights;
        shared_ptr<BaseTensor> last_input;
        bool use_32_bit;
        vector<vector<size_t>> input_shapes;
        vector<vector<size_t>> output_shapes;
        shared_ptr<MBGDLearningState> learning_state;
    };

    class SGDBias : public NeuralNetworkFunction {
    public:
        SGDBias(size_t input_size, size_t output_size, bool use_32_bit,
                const shared_ptr<MBGDLearningState> &learning_state) {
            this->input_shapes = vector<vector<size_t>>{{1, input_size, 1}};
            this->output_shapes = vector<vector<size_t>>{{1, output_size, 1}};
            this->bias = make_shared<TensorFromRandom>(input_size, output_size, 1, -0.5f, 0.5f, 42);
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
                bias = std::make_shared<FullTensor>(adjusted_bias);
//                cout << endl << "32 bias" << endl;
//                bias->print();
            } else {
                float adj_min = adjusted_bias->min();
                float adj_max = adjusted_bias->max();
                int result_bias = 8;
                for(int proposed_bias = 15; proposed_bias >= 9; proposed_bias--) {
                    float bias_max = quarter_to_float(QUARTER_MAX, proposed_bias, 0);
                    float bias_min = -bias_max;
                    if(adj_min > bias_min && adj_max < bias_max) {
                        result_bias = proposed_bias - 1;
                        break;
                    }
                }
                bias = std::make_shared<QuarterTensor>(adjusted_bias, result_bias, 0);

//                cout << endl << "adjusted bias:";
//                adjusted_bias->print();
//                cout << endl << "alternate bias:";
//                auto alternate_bias = make_shared<FullTensor>(*adjusted_bias);
//                alternate_bias->print();
//                cout << endl << "actual bias:";
//                bias->print();
            }

            last_input.reset();
//            auto adjusted_output = std::make_shared<TensorMinusTensorView>(output_error, adjusted_bias);
            // TODO: partial derivative of bias would always be 1, so we pass along original error. I'm fairly sure this is right.
            // but I notice that the quarter float doesn't handle big shifts in scale very well
            return output_error;
        }

    private:
        shared_ptr<BaseTensor> bias;
        shared_ptr<BaseTensor> last_input;
        bool use_32_bit;
        vector<vector<size_t>> input_shapes;
        vector<vector<size_t>> output_shapes;
        shared_ptr<MBGDLearningState> learning_state;
    };

    class SGDOptimizer : public Optimizer {
    public:
        explicit SGDOptimizer(float learning_rate) {
            this->sgdLearningState = make_shared<MBGDLearningState>();
            this->sgdLearningState->learning_rate = learning_rate;
        }

        shared_ptr<NeuralNetworkFunction>
        createFullyConnectedNeurons(size_t input_size, size_t output_size, bool use_32_bit) override {
            return make_shared<MBGDFullyConnectedNeurons>(input_size, output_size, use_32_bit, sgdLearningState);
        }

        shared_ptr<NeuralNetworkFunction> createBias(size_t input_size, size_t output_size, bool use_32_bit) override {
            return make_shared<SGDBias>(input_size, output_size, use_32_bit, sgdLearningState);
        }

    private:
        shared_ptr<MBGDLearningState> sgdLearningState;
    };
}
#endif //MICROML_MBGD_OPTIMIZER_HPP
