//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_ROTATE_180_VIEW_HPP
#define HAPPYML_TENSOR_ROTATE_180_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorRotate180View : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorRotate180View(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
            row_base_value = child->rowCount() - 1;
            column_base_value = child->columnCount() - 1;
        }

        void printMaterializationPlan() override {
            cout << "TensorRotate180View{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row_base_value - row, column_base_value - column, channel);
            return val;
        }

    private:
        size_t row_base_value;
        size_t column_base_value;
    };
}

#endif //HAPPYML_TENSOR_ROTATE_180_VIEW_HPP
