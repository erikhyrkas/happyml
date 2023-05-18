//
// Created by Erik Hyrkas on 11/5/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_ADAM_OPTIMIZER_HPP
#define HAPPYML_ADAM_OPTIMIZER_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include "../optimizer.hpp"
#include "../../types/tensor_impls/uniform_tensor.hpp"
#include "../../types/tensor_views/scalar_add_tensor_view.hpp"
#include "../../types/tensor_views/element_wise_multiply_tensor_view.hpp"
#include "../../types/tensor_views/add_tensor_view.hpp"
#include "../../util/tensor_utils.hpp"
#include "../../types/tensor_views/power_tensor_view.hpp"
#include "../../types/tensor_views/sqrt_tensor_view.hpp"
#include "../../types/tensor_views/element_wise_inverse_tensor_view.hpp"

using std::vector;
using std::unordered_map;

namespace happyml {

    class AdamOptimizer : public BaseOptimizer {
    public:
        // Note: useDecayMomentum = demon
        AdamOptimizer(float learningRate, float biasLearningRate, bool useDecayMomentum = true)
                : BaseOptimizer(learningRate, biasLearningRate), useDecayMomentum(useDecayMomentum), next_id(0) {
            // Initialize hyperparameters
            beta1 = 0.9;
            beta2 = 0.999;
            epsilon = 1e-8;
        }

        int registerForWeightChanges() override {
            int id = next_id++;
            return id;
        }

        int registerForBiasChanges() override {
            int id = next_id++;
            return id;
        }

        shared_ptr<BaseTensor> calculateWeightsChange(int registration_id,
                                                      const shared_ptr<BaseTensor> &weights,
                                                      const shared_ptr<BaseTensor> &loss_gradient) override {
            PROFILE_BLOCK(profileBlock);

            // Initialize if needed
            if (weight_m.count(registration_id) == 0) {
                // initialize to tensors of all 0s
                auto immutable_zero_tensor = make_shared<UniformTensor>(weights->getShape(), 0.0f);
                weight_m[registration_id] = immutable_zero_tensor;
                weight_v[registration_id] = immutable_zero_tensor;
            }
            if (useDecayMomentum && time_step > last_time_step_updated_weights) {
                last_time_step_updated_weights = time_step;
                learningRate = calculateDemonAdjustedLearnRate(weight_m, weight_v);
            }
            // Adam optimizer update rule
            return adam_update(registration_id, weights, loss_gradient,
                               weight_m, weight_v, learningRate);
        }

        shared_ptr<BaseTensor> calculateBiasChange(int registration_id, const shared_ptr<BaseTensor> &bias,
                                                   const shared_ptr<BaseTensor> &loss_gradient) override {
            PROFILE_BLOCK(profileBlock);

            // Initialize if needed
            if (bias_m.count(registration_id) == 0) {
                // initialize to tensors of all 0s
                auto immutable_zero_tensor = make_shared<UniformTensor>(bias->getShape(), 0.0f);
                bias_m[registration_id] = immutable_zero_tensor;
                bias_v[registration_id] = immutable_zero_tensor;
            }
            if (useDecayMomentum && time_step > last_time_step_updated_bias) {
                last_time_step_updated_bias = time_step;
                biasLearningRate = calculateDemonAdjustedLearnRate(bias_m, bias_v);
            }
            // Adam optimizer update rule
            return adam_update(registration_id, bias, loss_gradient,
                               bias_m, bias_v, biasLearningRate);
        }


    private:
        bool useDecayMomentum;
        int next_id;
        float beta1, beta2, epsilon;
        const float smallest_learning_rate = 1e-5;
        const float largest_learning_rate = 1e-1;
        int last_time_step_updated_bias = 1;
        int last_time_step_updated_weights = 1;
        unordered_map<int, shared_ptr<BaseTensor>> weight_m, weight_v;
        unordered_map<int, shared_ptr<BaseTensor>> bias_m, bias_v;

        float calculateDemonAdjustedLearnRate(const unordered_map<int, shared_ptr<BaseTensor>> &m_map,
                                              const unordered_map<int, shared_ptr<BaseTensor>> &v_map) {
            PROFILE_BLOCK(profileBlock);

            // time_step changes once per epoch
            // DEMON
            float m_average = average(m_map);
            float v_average = average(v_map);

            // Compute bias-corrected first moment estimate
            float beta1_pow = std::powf(beta1, (float) time_step);
            float temp_inverse_complement_beta1_pow_time_step = min(largest_learning_rate,
                                                                    max(smallest_learning_rate,
                                                                        1.0f / (1.0f - beta1_pow)));
            float m_hat_average = m_average * temp_inverse_complement_beta1_pow_time_step;

            // Compute bias-corrected second raw moment estimate
            float beta2_pow = std::powf(beta2, (float) time_step);
            float temp_inverse_complement_beta2_pow_time_step = 1.0f / (1.0f - beta2_pow);
            float v_hat_average = v_average * temp_inverse_complement_beta2_pow_time_step;

            float demon = m_hat_average / (std::sqrt(v_hat_average) + epsilon);
            return std::min(largest_learning_rate, std::max(smallest_learning_rate, demon));
        }

        shared_ptr<BaseTensor> adam_update(int registration_id,
                                           const shared_ptr<BaseTensor> &params,
                                           const shared_ptr<BaseTensor> &loss_gradient,
                                           unordered_map<int, shared_ptr<BaseTensor>> &m_map,
                                           unordered_map<int, shared_ptr<BaseTensor>> &v_map,
                                           float lr) {
            PROFILE_BLOCK(profileBlock);

            // Update biased first moment estimate
            auto temp_m_by_beta1 = make_shared<ScalarMultiplyTensorView>(m_map[registration_id], beta1);
            auto temp_beta_complement_value = make_shared<ScalarMultiplyTensorView>(loss_gradient, 1 - beta1);
            auto biased_first_moment_estimate = make_shared<AddTensorView>(temp_m_by_beta1,
                                                                           temp_beta_complement_value);

            // Update biased second raw moment estimate
            auto temp_v_by_beta2 = make_shared<ScalarMultiplyTensorView>(v_map[registration_id], beta2);
            auto temp_loss_pow_value = make_shared<PowerTensorView>(loss_gradient, 2.0f);
            auto temp_beta2_complement_value = make_shared<ScalarMultiplyTensorView>(temp_loss_pow_value, 1 - beta2);
            auto biased_second_moment_estimate = make_shared<AddTensorView>(temp_v_by_beta2,
                                                                            temp_beta2_complement_value);

            shared_ptr<BaseTensor> m_hat = materializeTensor(biased_first_moment_estimate);
            shared_ptr<BaseTensor> v_hat = materializeTensor(biased_second_moment_estimate);
            m_map[registration_id] = m_hat;
            v_map[registration_id] = v_hat;

            // Update parameters
            auto temp_lr_by_m_hat = make_shared<ScalarMultiplyTensorView>(m_hat,
                                                                          -lr); // make negative so we can subtract later
            auto temp_sqrt = make_shared<ScalarAddTensorView>(make_shared<SqrtTensorView>(v_hat), epsilon);
            auto temp_inverse_sqrt = make_shared<ElementWiseInverseTensorView>(temp_sqrt);
            auto temp_elementwise_product = make_shared<ElementWiseMultiplyTensorView>(temp_lr_by_m_hat, temp_inverse_sqrt);
            auto updated_params = make_shared<AddTensorView>(params,
                                                             temp_elementwise_product); // now the negative lr makes sense.

            return updated_params;
        }

        static float average(const unordered_map<int, shared_ptr<BaseTensor>> &tensors) {
            PROFILE_BLOCK(profileBlock);
            float sum_of_averages = 0;

            for (const auto &[key, tensor]: tensors) {
                float tensor_mean = tensor->arithmeticMean();
                sum_of_averages += tensor_mean;
            }

            return sum_of_averages / (float) tensors.size();
        }
    };
}
#endif //HAPPYML_ADAM_OPTIMIZER_HPP
