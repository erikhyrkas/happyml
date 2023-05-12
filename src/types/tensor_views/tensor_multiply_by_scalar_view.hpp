//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_MULTIPLY_BY_SCALAR_VIEW_HPP
#define HAPPYML_TENSOR_MULTIPLY_BY_SCALAR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// Multiply each element of the tensor by a constant.
    class TensorMultiplyByScalarView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorMultiplyByScalarView(const shared_ptr<BaseTensor> &tensor, float scale) : BaseTensorUnaryOperatorView(
                tensor) {
            this->scale = scale;
        }

        void printMaterializationPlan() override {
            cout << "TensorMultiplyByScalarView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return scale * child_->getValue(row, column, channel);
        }

        [[nodiscard]] float get_scale() const {
            return scale;
        }

    private:
        float scale;
    };
}
#endif //HAPPYML_TENSOR_MULTIPLY_BY_SCALAR_VIEW_HPP
