//
// Created by Erik Hyrkas on 4/8/2023.
//

#ifndef HAPPYML_SGDM_OPTIMIZER_HPP
#define HAPPYML_SGDM_OPTIMIZER_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include "optimizer.hpp"

using namespace std;

namespace happyml {
    class SGDMOptimizer : public BaseOptimizer {
    public:
        SGDMOptimizer(float learningRate, float biasLearningRate, bool use_decay_momentum = true)
                : BaseOptimizer(learningRate, biasLearningRate), use_decay_momentum_(use_decay_momentum),
                  next_id(0) {
            // Initialize hyperparameters
            momentumFactor = 0.9;
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
            // Initialize if needed
            if (weight_momentum.count(registration_id) == 0) {
                auto zero_tensor = make_shared<UniformTensor>(weights->rowCount(), weights->columnCount(),
                                                              weights->channelCount(), 0.0f);
                weight_momentum[registration_id] = zero_tensor;
            }

            // Momentum update rule
            auto prev_momentum = weight_momentum[registration_id];
            shared_ptr<BaseTensor> new_momentum = make_shared<TensorMultiplyByScalarView>(prev_momentum,
                                                                                          momentumFactor);
            auto scaled_gradient = make_shared<TensorMultiplyByScalarView>(loss_gradient, learningRate);
            new_momentum = make_shared<TensorAddTensorView>(new_momentum, scaled_gradient);
            weight_momentum[registration_id] = materializeTensor(new_momentum);

            auto updated_weights = make_shared<TensorMinusTensorView>(weights,
                                                                                        weight_momentum[registration_id]);

            if (use_decay_momentum_ && time_step > last_time_step_updated_weights) {
                last_time_step_updated_weights = time_step;
                learningRate = calculateDemonAdjustedLearnRate(weight_momentum);
            }

            return updated_weights;
        }

        shared_ptr<BaseTensor> calculateBiasChange(int registration_id,
                                                   const shared_ptr<BaseTensor> &bias,
                                                   const shared_ptr<BaseTensor> &loss_gradient) override {
            // Initialize if needed
            if (bias_momentum.count(registration_id) == 0) {
                auto zero_tensor = make_shared<UniformTensor>(bias->rowCount(), bias->columnCount(),
                                                              bias->channelCount(), 0.0f);
                bias_momentum[registration_id] = zero_tensor;
            }

            // Momentum update rule
            auto prev_momentum = bias_momentum[registration_id];
            shared_ptr<BaseTensor> new_momentum = make_shared<TensorMultiplyByScalarView>(prev_momentum,
                                                                                          momentumFactor);
            auto scaled_gradient = make_shared<TensorMultiplyByScalarView>(loss_gradient, biasLearningRate);
            new_momentum = make_shared<TensorAddTensorView>(new_momentum, scaled_gradient);
            bias_momentum[registration_id] = materializeTensor(new_momentum);

            auto updated_bias = make_shared<TensorMinusTensorView>(bias, bias_momentum[registration_id]);

            if (use_decay_momentum_ && time_step > last_time_step_updated_bias) {
                last_time_step_updated_bias = time_step;
                biasLearningRate = calculateDemonAdjustedLearnRate(bias_momentum);
            }

            return updated_bias;
        }

    private:
        bool use_decay_momentum_;
        int next_id;
        int last_time_step_updated_bias = 1;
        int last_time_step_updated_weights = 1;
        float momentumFactor, beta1, beta2, epsilon;
        const float smallest_learning_rate = 1e-5;
        const float largest_learning_rate = 1e-1;
        unordered_map<int, shared_ptr<BaseTensor>> weight_momentum;
        unordered_map<int, shared_ptr<BaseTensor>> bias_momentum;

        float calculateDemonAdjustedLearnRate(const unordered_map<int, shared_ptr<BaseTensor>> &momentum_map) {
            // DEMON
            float m_average = average(momentum_map);

            // Compute bias-corrected first moment estimate
            float beta1_pow = std::powf(beta1, (float) time_step);
            float temp_inverse_complement_beta1_pow_time_step = min(largest_learning_rate,
                                                                    max(smallest_learning_rate,
                                                                        1.0f / (1.0f - beta1_pow)));
            float m_hat_average = m_average * temp_inverse_complement_beta1_pow_time_step;

            // Compute bias-corrected second raw moment estimate
            float beta2_pow = std::powf(beta2, (float) time_step);
            float temp_inverse_complement_beta2_pow_time_step = 1.0f / (1.0f - beta2_pow);
            float v_hat_average = m_average * m_average * temp_inverse_complement_beta2_pow_time_step;

            float demon = m_hat_average / (std::sqrt(v_hat_average) + epsilon);
            return std::min(largest_learning_rate, std::max(smallest_learning_rate, demon));
        }

        static float average(const unordered_map<int, shared_ptr<BaseTensor>> &tensors) {
            float sum_of_averages = 0;

            for (const auto &[key, tensor]: tensors) {
                float tensor_mean = tensor->arithmeticMean();
                sum_of_averages += tensor_mean;
            }

            return sum_of_averages / (float) tensors.size();
        }
    };
}
#endif //HAPPYML_SGDM_OPTIMIZER_HPP
