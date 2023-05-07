//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_ELEMENT_WISE_INVERSE_VIEW_HPP
#define HAPPYML_TENSOR_ELEMENT_WISE_INVERSE_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    // returns inverse of each value( 1.0f / original value )
    class TensorElementWiseInverseView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorElementWiseInverseView(const std::shared_ptr<BaseTensor> &tensor, float epsilon = 1e-8) : BaseTensorUnaryOperatorView(
                tensor) {
            this->epsilon = epsilon;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel) + epsilon;
            return 1.0f / val;
        }

        void printMaterializationPlan() override {
            cout << "TensorElementWiseInverseView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

    private:
        float epsilon;
    };
}

#endif //HAPPYML_TENSOR_ELEMENT_WISE_INVERSE_VIEW_HPP
