//
// Created by Erik Hyrkas on 10/29/2022.
//

#ifndef MICROML_LAYER_HPP
#define MICROML_LAYER_HPP

#include "tensor.hpp"
#include "activation.hpp"

class Layer {
public:
    virtual size_t number_of_outputs() = 0;

    // we take in an input, but we may have 1 or more outputs
    virtual std::vector<BaseTensor> infer(BaseTensor input) = 0;

    // TODO: this is probably not right -> need to update weights and bias during backward propagation, but the shape of this method probably needs to change
    virtual std::vector<BaseTensor> train(BaseTensor input) = 0;
};

// I want the "standard layer" to be an ActivatableLayer that takes in an activation function and specifies whether to do batch normalization.
// PyTorch has a linear layer that you then activate manually and Tensorflow has a dense layer that takes in an activation function, but
// it doesn't have built in batch normalization.
// regularizer l1/l2
// bool batchNormalization
class ActivatableLayer : public Layer {
public:
    ActivatableLayer(std::shared_ptr<ActivationFunction> activationFunction, size_t neurons) {
        this->activationFunction = activationFunction;
    }

private:
    std::shared_ptr<ActivationFunction> activationFunction;

};

class BiasActivatableLayer : public ActivatableLayer {
public:
    BiasActivatableLayer(std::shared_ptr<ActivationFunction> activationFunction, size_t neurons) : ActivatableLayer(activationFunction, neurons){
        this->activationFunction = activationFunction;
    }

private:
    std::shared_ptr<ActivationFunction> activationFunction;
    float bias_value;
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
