//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_MATRIX_DIVIDE_TENSOR_VIEW_HPP
#define HAPPYML_MATRIX_DIVIDE_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>
#include "inverse_tensor_view.hpp"

namespace happyml {
    // matdiv
    class MatrixDivideTensorView : public BaseTensorBinaryOperatorView {
    public:
        MatrixDivideTensorView(const shared_ptr<BaseTensor> &left_child,
                               const shared_ptr<BaseTensor> &right_child) : BaseTensorBinaryOperatorView(left_child,
                                                                                                               make_shared<InverseTensorView>(right_child)) {
            if (left_child_->rowCount() != right_child_->rowCount() || left_child_->columnCount() != right_child_->columnCount()) {
                string error_message = "Matrix dimensions must match for division. Left: " + to_string(left_child_->rowCount()) + "x" + to_string(left_child_->columnCount()) + " Right: " + to_string(right_child_->rowCount()) + "x" + to_string(right_child_->columnCount());
                throw runtime_error(error_message);
            }
            if (left_child_->channelCount() != right_child_->channelCount()) {
                string error_message = "Channel count must match for division. Left: " + to_string(left_child_->channelCount()) + " Right: " + to_string(right_child_->channelCount());
                throw runtime_error(error_message);
            }
        }

        void printMaterializationPlan() override {
            cout << "MatrixDivideTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") / (";
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
            // Compute the inverse of the right_child_ matrix for the given row, column, channel
            float val = 0;
            for (size_t col = 0; col < left_child_->columnCount(); col++) {
                val += left_child_->getValue(row, col, channel) * right_child_->getValue(row, col, channel);
            }
            return val;
        }
    };
}

#endif //HAPPYML_MATRIX_DIVIDE_TENSOR_VIEW_HPP
