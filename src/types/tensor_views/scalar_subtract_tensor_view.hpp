//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_SCALAR_SUBTRACT_TENSOR_VIEW_HPP
#define HAPPYML_SCALAR_SUBTRACT_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {

    // Subtracts a constant to every value of a matrix through a view
    class ScalarSubtractTensorView : public happyml::BaseTensorUnaryOperatorView {
    public:
        ScalarSubtractTensorView(float adjustment, const shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
            this->inverted = true;
        }

        ScalarSubtractTensorView(const shared_ptr<BaseTensor> &tensor, float adjustment)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
            this->inverted = false;
        }

        void printMaterializationPlan() override {
            cout << "SubtractTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
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
#endif //HAPPYML_SCALAR_SUBTRACT_TENSOR_VIEW_HPP
