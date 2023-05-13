//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_ROTATE_180_TENSOR_VIEW_HPP
#define HAPPYML_ROTATE_180_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class Rotate180TensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit Rotate180TensorView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
            row_base_value = child_->rowCount() - 1;
            column_base_value = child_->columnCount() - 1;
        }

        void printMaterializationPlan() override {
            cout << "Rotate180TensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row_base_value - row, column_base_value - column, channel);
            return val;
        }

    private:
        size_t row_base_value;
        size_t column_base_value;
    };
}

#endif //HAPPYML_ROTATE_180_TENSOR_VIEW_HPP
