//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
    // matmul
    class TensorMatrixMultiplyTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorMatrixMultiplyTensorView(const shared_ptr<BaseTensor> &tensor1,
                                       const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1,
                                                                                                        tensor2) {
            if (tensor1->columnCount() != tensor2->rowCount()) {
                cout << "[" << tensor1->rowCount() << ", " << tensor1->columnCount() << ", " << tensor1->channelCount()
                     << "] * [";
                cout << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                     << endl;
                throw runtime_error("matmul tensor1.cols must match tensor2.rows in length");
            }
            if (tensor1->channelCount() != tensor2->channelCount()) {
                throw runtime_error("matmul tensor1.channels must match tensor2.channels in length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorMatrixMultiplyTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") * (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return left_child_->rowCount();
        }

        size_t columnCount() override {
            return right_child_->columnCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            float val = 0;
            const auto childColumnCount = left_child_->columnCount();
//#pragma omp for
            for (long long t1_col = 0; t1_col < childColumnCount; t1_col++) {
                val += left_child_->getValue(row, t1_col, channel) * right_child_->getValue(t1_col, column, channel);
            }
            return val;
        }
    };
}

#endif //HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP
