//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_BIAS_LAYER_HPP
#define HAPPYML_BIAS_LAYER_HPP

#include "../types/tensor_views/scalar_divide_tensor_view.hpp"

namespace happyml {
    class BiasLayer : public BaseLayer {
    public:
        BiasLayer(const string &label, const vector<size_t> &inputShape, const vector<size_t> &outputShape,
                  uint8_t bits,
                  int optimizer_registration_id) : bias_errors() {
            this->label = label;
            this->registration_id = optimizer_registration_id;
            this->inputShapes = vector<vector<size_t >>{inputShape};
            this->outputShape = outputShape;
            this->bias = make_shared<TensorFromXavier>(outputShape[0], outputShape[1], outputShape[2], 42);
            this->bits = bits;
        }

        vector<vector<size_t>> getInputShapes() {
            return inputShapes;
        }

        vector<size_t> getOutputShape() {
            return outputShape;
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            bias->save(path);
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            this->bias = make_shared<FullTensor>(path);
        }

        // predicting
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw runtime_error("BiasNeuron only supports a single input.");
            }

            return make_shared<AddTensorView>(input[0], bias);
        }

        // learning
        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);

            bias_errors.push(output_error);

            // TODO: partial derivative of bias would always be 1, so we pass along original error. I'm fairly sure this is right.
            //  but I notice that the quarter float doesn't handle big shifts in scale very well. Part of me thinks
            //  that bias should always be a full float, but I'm not sure. Maybe a half-float is fine.
            return {output_error};
        }

        void apply(const shared_ptr<BaseOptimizer> &optimizer) override {
            PROFILE_BLOCK(profileBlock);

            if (bias_errors.empty()) {
                throw runtime_error("BiasNeuron::apply() called without any errors having been pushed.");
            }
            size_t current_batch_size = bias_errors.size();
            shared_ptr<BaseTensor> output_error = bias_errors.front();
            bias_errors.pop();
            while (!bias_errors.empty()) {
                output_error = make_shared<AddTensorView>(output_error, bias_errors.front());
                bias_errors.pop();
            }
            output_error = materializeTensor(make_shared<ScalarDivideTensorView>(output_error, current_batch_size));

            auto adjusted_bias = optimizer->calculateBiasChange(registration_id,
                                                                bias,
                                                                output_error);
            bias = materializeTensor(adjusted_bias, bits);
        }

        bool is_trainable() override {
            return true;
        }

        size_t get_parameter_count() override {
            return bias->size();
        }
    private:
        int registration_id;
        shared_ptr<BaseTensor> bias;
        queue<shared_ptr<BaseTensor>> bias_errors;
        uint8_t bits;
        vector<vector<size_t>> inputShapes;
        vector<size_t> outputShape;
        string label;
    };
}
#endif //HAPPYML_BIAS_LAYER_HPP
