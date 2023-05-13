//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_VALID_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP
#define HAPPYML_VALID_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class Valid2DCrossCorrelationTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        Valid2DCrossCorrelationTensorView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : BaseTensorBinaryOperatorView(tensor, kernel) {
            rows = left_child_->rowCount() - right_child_->rowCount() + 1;
            cols = left_child_->columnCount() - right_child_->columnCount() + 1;
        }

        void printMaterializationPlan() override {
            cout << "Valid2DCrossCorrelationTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") + (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const auto kernel_rows = right_child_->rowCount();
            const auto kernel_cols = right_child_->columnCount();
            float result = 0.f;
//#pragma omp for collapse(2)
            for (long long kernel_row = 0; kernel_row < kernel_rows; kernel_row++) {
                for (long long kernel_col = 0; kernel_col < kernel_cols; kernel_col++) {
                    const auto kernel_val = right_child_->getValue(kernel_row, kernel_col,
                                                                   channel); // channel 0 is applied to all channels of tensor
                    const auto tensor_val = left_child_->getValue(row + kernel_row, column + kernel_col, channel);
                    result += kernel_val * tensor_val;
                }
            }
            return result;
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return cols;
        }

        size_t channelCount() override {
            return left_child_->channelCount();
        }

    private:
        size_t rows;
        size_t cols;
    };
}

#endif //HAPPYML_VALID_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP
