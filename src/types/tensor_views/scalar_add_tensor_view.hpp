//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SCALAR_ADD_TENSOR_VIEW_HPP
#define HAPPYML_SCALAR_ADD_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// Adds a constant to every value of a matrix through a view
    class ScalarAddTensorView : public BaseTensorUnaryOperatorView {
    public:
        ScalarAddTensorView(const shared_ptr<BaseTensor> &tensor, float adjustment)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
        }

        void printMaterializationPlan() override {
            cout << "ScalarAddTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return child_->getValue(row, column, channel) + adjustment;
        }

        [[nodiscard]] float get_adjustment() const {
            return adjustment;
        }

    private:
        float adjustment;
    };
}

#endif //HAPPYML_SCALAR_ADD_TENSOR_VIEW_HPP
