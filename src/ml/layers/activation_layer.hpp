//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_ACTIVATION_LAYER_HPP
#define HAPPYML_ACTIVATION_LAYER_HPP


namespace happyml {
    class ActivationLayer : public NeuralNetworkLayerFunction {
    public:
        explicit ActivationLayer(const shared_ptr<ActivationFunction> &activationFunction) {
            this->activationFunction = activationFunction;
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            // todo: throw error on wrong size input?
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw runtime_error("Cannot activate multiple inputs at the same time. Please merge.");
            }
            const auto &lastInput = input[0];
            if (forTraining) {
                lastInputs.push(lastInput);
            }
            return activationFunction->activate(lastInput);
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &outputError) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize < 1) {
                throw runtime_error("backward() called without previous inputs.");
            }

            shared_ptr<BaseTensor> averageActivationDerivative = activationFunction->derivative(lastInputs.front());
            lastInputs.pop();
            if (lastInputsSize > 1) {
                while (!lastInputs.empty()) {
                    auto nextLastInput = activationFunction->derivative(lastInputs.front());
                    lastInputs.pop();
                    averageActivationDerivative = make_shared<AddTensorView>(averageActivationDerivative,
                                                                             nextLastInput);
                }
                averageActivationDerivative = materializeTensor(
                        make_shared<ScalarDivideTensorView>(averageActivationDerivative, (float) lastInputsSize));
            }

            //auto activation_derivative = activationFunction->derivative(average_last_inputs);
            // this really threw me for a loop. I thought that this was supposed to be dot product, rather than
            // an element-wise-multiplication.
            const auto baseOutputError = make_shared<ElementWiseMultiplyTensorView>(averageActivationDerivative,
                                                                                    outputError);
            return {baseOutputError};
        }

    private:
        shared_ptr<ActivationFunction> activationFunction;
        queue<shared_ptr<BaseTensor>> lastInputs; // each input in a batch will queue in order during forward, and deque properly when back-propagating
    };
}
#endif //HAPPYML_ACTIVATION_LAYER_HPP
