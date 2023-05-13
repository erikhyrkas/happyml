//
// Created by Erik Hyrkas on 5/12/2022.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_DIVIDE_BY_SCALAR_VIEW_HPP
#define HAPPYML_TENSOR_DIVIDE_BY_SCALAR_VIEW_HPP

#include <sstream>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    // Divide each element of the tensor by a constant.
    class ScalarDivideTensorView : public BaseTensorUnaryOperatorView {
    public:
        ScalarDivideTensorView(const shared_ptr<BaseTensor> &tensor, float denominator)
                : BaseTensorUnaryOperatorView(tensor) {
            this->val_ = denominator;
            scalar_is_denominator_ = true;
        }
        ScalarDivideTensorView(float numerator, const shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
            this->val_ = numerator;
            scalar_is_denominator_ = false;
        }
        void printMaterializationPlan() override {
            cout << "ScalarDivideTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (scalar_is_denominator_) {
                return child_->getValue(row, column, channel) / val_;
            }
            return val_ / child_->getValue(row, column, channel);
        }

        [[nodiscard]] float get_denominator() const {
            return val_;
        }

    private:
        float val_;
        float scalar_is_denominator_;
    };
}
#endif //HAPPYML_TENSOR_DIVIDE_BY_SCALAR_VIEW_HPP
