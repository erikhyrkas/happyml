//
// Created by Erik Hyrkas on 5/15/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_STANDARDIZE_DERIVITIVE_TENSOR_VIEW_HPP
#define HAPPYML_STANDARDIZE_DERIVITIVE_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <utility>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {

    class StandardizeDerivativeTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit StandardizeDerivativeTensorView(const shared_ptr<BaseTensor> &d_output, const shared_ptr<BaseTensor> &input, float mean, float std_dev)
                : BaseTensorUnaryOperatorView(d_output), mean_(mean), std_dev_(std_dev), input_(input) {
            total_elements_ = input->size();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            // get corresponding values
            float d_output_value = child_->getValue(row, column, channel);
            float input_value = input_->getValue(row, column, channel);

            // calculate common part
            float common = d_output_value / (std_dev_ * (float) total_elements_);

            // calculate three parts of gradient
            float dx1 = d_output_value / std_dev_;
            float dx2 = -common * (input_value - mean_);
            float dx3 = -2.0f / (float) total_elements_ * (input_value - mean_) * common;

            // return total gradient
            return dx1 + dx2 + dx3;
        }

        void printMaterializationPlan() override {
            cout << "StandardizeDerivativeTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    private:
        float mean_;
        float std_dev_;
        shared_ptr<BaseTensor> input_;
        size_t total_elements_;
    };
}

#endif //HAPPYML_STANDARDIZE_DERIVITIVE_TENSOR_VIEW_HPP
