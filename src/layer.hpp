//
// Created by Erik Hyrkas on 10/29/2022.
//

#ifndef MICROML_LAYER_HPP
#define MICROML_LAYER_HPP

#include "tensor.hpp"
#include "activation.hpp"
// TODO: I'll need an abstraction to allow multiple inputs or multiple outputs
// NOTES:
// * Neural networks are usually thought of in layers, but you can have two "layers" in parallel. This makes the
//   term confusing. Really, Neural networks are ... networks... and while simple neural networks are very linear,
//   there are many situations where we have multiple outputs and points in the neural networks were we have nodes
//   that run in parallel and then need to be merged. The term "layer" is mostly bad, confusing, and not useful.
// * the only real multiple output situation that i can think of is for the final step, which is not really a layer.
//   We just have more than one layer at that step.
// * the multiple input situation is where we have a step that needs to merge two layers
// * A "neural network" is really a directed acyclic graph made up of nodes not layers... but making a graph of nodes with multiple
//   inputs and multiple outputs and tracking which nodes to run at which time is complicated. I've done it and I started to
//   do it for this project...but I got to thinking: is there another way to represent a dag that's easy to visualize and use?
// * The answer I came to is: containers. You can have a container object that holds other containers to whichever granularity
//   until you get to the point where you have logic that does something.
// * A container like this isn't a perfect representation for every single type of network one could imagine. After all,
//   a single node can only ever exist in one container. However, I think we can still use this representation -- what's
//   more, I think this representation helps us void running afoul of issues we might otherwise have with tensor views. It
//   doesn't guarantee it, which is one of the reasons I didn't like allowing tensors to be assignable, but compromises must be made.
//
// Conclusion:
// * I'm going to have a Node (or Network Part) -- which is a container object representing a portion of a neural network.
// * Network parts take in a single input and have a single output
// * There are five types of network parts:
//   * Input Node -- reads from a data source
//   * Output Capture Node -- the input is stored as a materialized tensor
//   * Serial Node -- all children are processed in order. First child takes the network part's input and then the output of each part being the input to the next in line. Last child's output is this network part's output.
//   * Parallel Node -- all children are run in an arbitrary order (potentially in parallel) and merged to produce one output
//   * Neurons -- objects that learn (includes convolutional and fully connected representations)
// * Network parts need the ability to check for cycles. bool cycle_check(queue<shared_ptr<NetworkPart>> &parents)
// * I expect that some optimizers will want to wrap nodes to store their own state. I'm looking at you Adam.


// Something that learns
class Neuron {
public:
    virtual size_t number_of_outputs() = 0;

    virtual size_t number_of_inputs() = 0;

    // one input and one output.
    virtual std::shared_ptr<BaseTensor> forward(const std::shared_ptr<BaseTensor> &input) = 0;

    virtual std::shared_ptr<BaseTensor>
    backward(const std::shared_ptr<BaseTensor> &output_error, float learning_rate) = 0;
};

class BaseGraphNode {


private:
//    std::vector<std::shared_ptr<NetworkNode>> children;
};

// I want the "standard layer" to be an ActivatableLayer that takes in an activation function and specifies whether to do batch normalization.
// PyTorch has a linear layer that you then activate manually and Tensorflow has a dense layer that takes in an activation function, but
// it doesn't have built in batch normalization.
// regularizer l1/l2
// bool batchNormalization
class ActivatableNeuron : public Neuron {
public:
    ActivatableNeuron(const std::shared_ptr<ActivationFunction> &activationFunction, size_t input_size,
                      size_t output_size, bool use_32_bit) {
        this->activationFunction = activationFunction;
        this->input_size = input_size;
        this->output_size = output_size;
        this->weights = std::make_shared<TensorFromRandom>(input_size, output_size, 1, 14);
        this->use_32_bit = use_32_bit;
    }

    // predicting
    std::shared_ptr<BaseTensor> forward(const std::shared_ptr<BaseTensor> &input) override {
        last_input = input;
        last_unactivated_result = std::make_shared<TensorDotTensorView>(input, weights);
        return activationFunction->activate(last_unactivated_result);
    }

    // learning
    std::shared_ptr<BaseTensor>
    backward(const std::shared_ptr<BaseTensor> &output_error, float learning_rate) override {
        // The output error is looking at the activated result, we need to find the error on the unactivated result.
        auto activation_derivative = activationFunction->derivative(last_unactivated_result);
        auto base_output_error = std::make_shared<TensorDotTensorView>(activation_derivative, output_error);

        // find the error
        auto weights_transposed = std::make_shared<TensorTransposeView>(weights);
        auto input_error = std::make_shared<TensorDotTensorView>(base_output_error, weights_transposed);

        // update weights
        auto input_transposed = std::make_shared<TensorTransposeView>(last_input);
        auto weights_error = std::make_shared<TensorDotTensorView>(input_transposed, base_output_error);
        auto weights_error_at_learning_rate = std::make_shared<TensorMultiplyByScalarView>(weights_error,
                                                                                           learning_rate);
        auto adjusted_weights = std::make_shared<TensorMinusTensorView>(weights, weights_error_at_learning_rate);
        if (use_32_bit) {
            weights = std::make_shared<FullTensor>(*adjusted_weights);
        } else {
            weights = std::make_shared<QuarterTensor>(*adjusted_weights, 14, 0);
        }

        // TODO: We might be able to free memory here now. Depends on whether backward is called multiple times.
        // last_input.reset();
        // last_unactivated_result.reset();
        return input_error;
    }

    size_t number_of_outputs() override {
        return output_size;
    }

    size_t number_of_inputs() override {
        return input_size;
    }

protected:
    std::shared_ptr<BaseTensor> weights;
    std::shared_ptr<BaseTensor> last_input;
    std::shared_ptr<BaseTensor> last_unactivated_result;
    bool use_32_bit;
private:
    std::shared_ptr<ActivationFunction> activationFunction;
    size_t input_size;
    size_t output_size;
};

// TODO: It's possible to achieve some code reuse, with a refactor where the activation and pre-activate logic are in separate functions.
// I'll worry about that later when I'm ready to swear about tech debt.
// A single BiasActivatableLayer is enough to do linear regression - just use LinearActivationFunction
class BiasActivatableNeuron : public ActivatableNeuron {
public:
    BiasActivatableNeuron(const std::shared_ptr<ActivationFunction> &activationFunction, size_t input_size,
                          size_t output_size, bool use_32_bit) : ActivatableNeuron(activationFunction, input_size,
                                                                                   output_size, use_32_bit) {
        this->activationFunction = activationFunction;
        this->bias_value = std::make_shared<TensorFromRandom>(input_size, output_size, 1, 14);
    }

    std::shared_ptr<BaseTensor> forward(const std::shared_ptr<BaseTensor> &input) override {
        // we can't reuse the parent's code because we have to add bias in before activation
        last_input = input;
        auto unbiased_result = std::make_shared<TensorDotTensorView>(input, weights);
        last_unactivated_result = std::make_shared<TensorAddTensorView>(unbiased_result, bias_value);
        return activationFunction->activate(last_unactivated_result);

    }

    // learning
    std::shared_ptr<BaseTensor>
    backward(const std::shared_ptr<BaseTensor> &output_error, float learning_rate) override {
        // we can't reuse the parent's code because we need to know the base output error for finding bias error
        // The output error is looking at the activated result, we need to find the error on the unactivated result.
        auto activation_derivative = activationFunction->derivative(last_unactivated_result);
        auto base_output_error = std::make_shared<TensorDotTensorView>(activation_derivative, output_error);

        // find the error
        auto weights_transposed = std::make_shared<TensorTransposeView>(weights);
        auto input_error = std::make_shared<TensorDotTensorView>(base_output_error, weights_transposed);

        // update weights
        auto input_transposed = std::make_shared<TensorTransposeView>(last_input);
        auto weights_error = std::make_shared<TensorDotTensorView>(input_transposed, base_output_error);
        auto weights_error_at_learning_rate = std::make_shared<TensorMultiplyByScalarView>(weights_error,
                                                                                           learning_rate);
        auto adjusted_weights = std::make_shared<TensorMinusTensorView>(weights, weights_error_at_learning_rate);
        if (use_32_bit) {
            weights = std::make_shared<FullTensor>(*adjusted_weights);
        } else {
            weights = std::make_shared<QuarterTensor>(*adjusted_weights, 14, 0);
        }

        // update bias
        auto bias_error_at_learning_rate = std::make_shared<TensorMultiplyByScalarView>(base_output_error,
                                                                                        learning_rate);
        auto adjusted_bias = std::make_shared<TensorMinusTensorView>(bias_value, bias_error_at_learning_rate);
        if (use_32_bit) {
            bias_value = std::make_shared<FullTensor>(*adjusted_bias);
        } else {
            bias_value = std::make_shared<QuarterTensor>(*adjusted_bias, 14, 0);
        }

        return input_error;
    }

private:
    std::shared_ptr<ActivationFunction> activationFunction;
    std::shared_ptr<BaseTensor> bias_value;
};

// Definitely need an "add" layer, that literally adds the two input matrices together (using matrix addition). Matrices must be same size.
// there are other layers like "subtract" and "dot" that may make sense, but seem lower priority to me.

// May want a "concatenate" layer that merges two matrices together. I think concatenating two matrices of the same shape has clear rules.
// I would default to row-centric concatination: You reshape the matrix with fewer rows max(rows_a, rows_b) so it has the same rows as the other,
// and then append columns_b to columns_a. You'll end up with a bunch of 0s in the empty places.
// I'd probably want a ConcatenateColumnsLayer and a ConcatenateRowsLayer.
// There are cases where we ensure we have the same number of columns and then append rows. I could see that being useful if you are trying to
// minimize the number of 0s you create.

// Likely will want convolution layers (at least 2d, but maybe 1d and 3d) and a recurrent layer

#endif //MICROML_LAYER_HPP
