//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_VALID_CROSS_CORRELATION_2D_VIEW_HPP
#define HAPPYML_TENSOR_VALID_CROSS_CORRELATION_2D_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorValidCrossCorrelation2dView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorValidCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : BaseTensorBinaryOperatorView(tensor, kernel) {
            rows = child1->rowCount() - child2->rowCount() + 1;
            cols = child1->columnCount() - child2->columnCount() + 1;
        }

        void printMaterializationPlan() override {
            cout << "TensorValidCrossCorrelation2dView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const auto kernel_rows = child2->rowCount();
            const auto kernel_cols = child2->columnCount();
            float result = 0.f;
//#pragma omp for collapse(2)
            for (long long kernel_row = 0; kernel_row < kernel_rows; kernel_row++) {
                for (long long kernel_col = 0; kernel_col < kernel_cols; kernel_col++) {
                    const auto kernel_val = child2->getValue(kernel_row, kernel_col,
                                                             channel); // channel 0 is applied to all channels of tensor
                    const auto tensor_val = child1->getValue(row + kernel_row, column + kernel_col, channel);
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
            return child1->channelCount();
        }

    private:
        size_t rows;
        size_t cols;
    };
}

#endif //HAPPYML_TENSOR_VALID_CROSS_CORRELATION_2D_VIEW_HPP
