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
    class TensorDivideByScalarView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorDivideByScalarView(const shared_ptr<BaseTensor> &tensor, float denominator) : BaseTensorUnaryOperatorView(
                tensor) {
            this->denominator_ = denominator;
        }

        void printMaterializationPlan() override {
            cout << "TensorDivideByScalarView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return child_->getValue(row, column, channel) / denominator_;
        }

        [[nodiscard]] float get_denominator() const {
            return denominator_;
        }

    private:
        float denominator_;
    };
}
#endif //HAPPYML_TENSOR_DIVIDE_BY_SCALAR_VIEW_HPP
