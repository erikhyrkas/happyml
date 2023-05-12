//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_MINUS_SCALAR_VIEW_HPP
#define HAPPYML_TENSOR_MINUS_SCALAR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {

    // Subtracts a constant to every value of a matrix through a view
    class TensorMinusScalarView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorMinusScalarView(float adjustment, const shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
            this->inverted = true;
        }

        TensorMinusScalarView(const shared_ptr<BaseTensor> &tensor, float adjustment)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
            this->inverted = false;
        }

        void printMaterializationPlan() override {
            cout << "TensorMinusScalarView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (inverted) {
                return adjustment - child_->getValue(row, column, channel);
            } else {
                return child_->getValue(row, column, channel) - adjustment;
            }
        }

        [[nodiscard]] float get_adjustment() const {
            return adjustment;
        }

    private:
        float adjustment;
        bool inverted;
    };
}
#endif //HAPPYML_TENSOR_MINUS_SCALAR_VIEW_HPP
