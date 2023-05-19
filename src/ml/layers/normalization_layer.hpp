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
        explicit NormalizationLayer() {
            lastInput = nullptr;
        }

        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                // only layers that combine input, like the concatenate_wide_layer, should have more than one input
                throw runtime_error("NormalizationLayer only supports a single input.");
            }
            auto &inputTensor = input[0];

            shared_ptr<StandardizeTensorView> normTensor = make_shared<StandardizeTensorView>(inputTensor);
            if (forTraining) {
                lastInput = normTensor;
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
            if (lastInput == nullptr) {
                throw runtime_error("No inputs to backpropagate through.");
            }

            shared_ptr<BaseTensor> normDerivativeTensor = make_shared<StandardizeDerivativeTensorView>(output_error, lastInput, lastInput->get_mean(), lastInput->get_std_dev());
#ifdef DEBUG_TRAIN_NAN
            if (normDerivativeTensor->hasNaNOrInf()) {
                normDerivativeTensor->print();
                throw runtime_error("NaN or Inf found in NormalizationLayer.");
            }
#endif

            return {normDerivativeTensor};
        }

    private:
        shared_ptr<StandardizeTensorView> lastInput;
    };

}
#endif //HAPPYML_NORMALIZATION_LAYER_HPP
