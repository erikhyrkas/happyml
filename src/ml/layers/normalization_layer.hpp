//
// Created by Erik Hyrkas on 5/15/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_NORMALIZATION_LAYER_HPP
#define HAPPYML_NORMALIZATION_LAYER_HPP

#include "../base_layer.hpp"
#include "../../types/tensor_views/standardize_tensor_view.hpp"
#include "../../types/tensor_views/standardize_derivative_tensor_view.hpp"
#include "../../types/tensor_views/add_tensor_view.hpp"
#include "../../types/tensor_views/scalar_divide_tensor_view.hpp"
#include "../../util/tensor_utils.hpp"

namespace happyml {
    class NormalizationLayer : public BaseLayer {
    public:
        explicit NormalizationLayer() : lastInputs() {
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw runtime_error("NormalizationLayer only supports a single input.");
            }
            auto &inputTensor = input[0];

            shared_ptr<StandardizeTensorView> normTensor = make_shared<StandardizeTensorView>(inputTensor);
            if (forTraining) {
                lastInputs.push(normTensor);
            }
#ifdef DEBUG_TRAIN_NAN
            if (normTensor->hasNaNOrInf()) {
                input[0]->print();
                normTensor->print();
                throw runtime_error("NaN or Inf found in NormalizationLayer.");
            }
#endif
            return normTensor;
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            size_t lastInputsSize = lastInputs.size();
            if (lastInputsSize == 0) {
                throw runtime_error("No inputs to backpropagate through.");
            }

            shared_ptr<BaseTensor> average_last_inputs = lastInputs.front();
            lastInputs.pop();
            if (lastInputsSize > 1) {
                while (!lastInputs.empty()) {
                    auto nextLastInput = lastInputs.front();
                    lastInputs.pop();
                    shared_ptr<BaseTensor> normDerivativeTensor = make_shared<StandardizeDerivativeTensorView>(output_error, nextLastInput, nextLastInput->get_mean(), nextLastInput->get_std_dev());
                    average_last_inputs = make_shared<AddTensorView>(average_last_inputs, normDerivativeTensor);
                }
                average_last_inputs = materializeTensor(
                        make_shared<ScalarDivideTensorView>(average_last_inputs, (float) lastInputsSize));
            }
#ifdef DEBUG_TRAIN_NAN
            if (average_last_inputs->hasNaNOrInf()) {
                average_last_inputs->print();
                throw runtime_error("NaN or Inf found in NormalizationLayer.");
            }
#endif

            return {average_last_inputs};
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            // TODO: right now, we don't have any state, is that right?
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            // TODO: right now, we don't have any state, is that right?
        }

    private:
        queue<shared_ptr<StandardizeTensorView>> lastInputs;
    };

}
#endif //HAPPYML_NORMALIZATION_LAYER_HPP
