#include <utility>

//
// Created by Erik Hyrkas on 5/28/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_DROPOUT_LAYER_HPP
#define HAPPYML_DROPOUT_LAYER_HPP

namespace happyml {
    class DropoutLayer : public BaseLayer {
    public:
        explicit DropoutLayer(std::string label, std::vector<size_t> input_shape, float dropout_rate = 0.5)
                : label_(std::move(label)), input_shape_(std::move(input_shape)), dropout_rate_(dropout_rate) {
            if (dropout_rate < 0.0 || dropout_rate > 1.0) {
                throw runtime_error("DropoutLayer: dropout rate must be between 0 and 1.");
            }
            forward_scale_ = 1.0f / (1.0f - dropout_rate_);
            output_shape_ = input_shape_;
        }

        std::shared_ptr<BaseTensor> forward(const std::vector<std::shared_ptr<BaseTensor>> &input, bool forTraining) override {
            if (forTraining) {
                // Initialize dropout_mask_ with the same shape as input
                seed_seq_++;
                shared_ptr < BaseTensor > random_tensor = make_shared<TensorFromRandom>(input[0]->getShape(), 0.0f, 1.0f, seed_seq_);
                shared_ptr < BaseTensor > zeros = make_shared<UniformTensor>(input[0]->getShape(), 0.0f);
                shared_ptr < BaseTensor > ones = make_shared<UniformTensor>(input[0]->getShape(), 1.0f);

                // Populate dropout_mask_ with random values from a uniform distribution between 0 and 1
                // then set to 1 if value > dropout_rate_, else 0
                dropout_mask_ = make_shared<MaskedSelectTensorView>(random_tensor, ones, zeros, dropout_rate_);

                // Multiply input by dropout_mask_
                return make_shared<ElementWiseMultiplyTensorView>(input[0], dropout_mask_);
            }
            return make_shared<ElementWiseMultiplyTensorView>(input[0], make_shared<UniformTensor>(input[0]->getShape(), forward_scale_));
        }

        std::vector<std::shared_ptr<BaseTensor>> backward(const std::shared_ptr<BaseTensor> &output_error) override {
            // Multiply output_error by dropout_mask_
            return {make_shared<ElementWiseMultiplyTensorView>(output_error, dropout_mask_)};
        }

        std::vector<size_t> get_output_shape() {
            return output_shape_;
        }

    private:
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
        string label_;
        float dropout_rate_;
        float forward_scale_;
        shared_ptr <BaseTensor> dropout_mask_;
        uint32_t seed_seq_ = 0;
    };
}
#endif //HAPPYML_DROPOUT_LAYER_HPP
