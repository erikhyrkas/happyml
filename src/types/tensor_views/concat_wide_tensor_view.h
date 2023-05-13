//
// Created by Erik Hyrkas on 5/11/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CONCAT_WIDE_TENSOR_VIEW_H
#define HAPPYML_CONCAT_WIDE_TENSOR_VIEW_H

#include "../base_tensors.hpp"

namespace happyml {
    class ConcatWideTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        ConcatWideTensorView(const shared_ptr<BaseTensor> &tensor1,
                             const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->rowCount() != tensor2->rowCount()) {
                cout << "[" << tensor1->rowCount() << ", " << tensor1->columnCount() << ", " << tensor1->channelCount()
                     << "] + [";
                cout << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                     << endl;
                throw runtime_error("You can only concatenate two tensors with the same number of rows.");
            }
        }

        void printMaterializationPlan() override {
            cout << "ConcatWideTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") + (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return left_child_->rowCount();
        }

        size_t columnCount() override {
            return left_child_->columnCount() + right_child_->columnCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (column < left_child_->columnCount()) {
                return left_child_->getValue(row, column, channel);
            } else {
                return right_child_->getValue(row, column - left_child_->columnCount(), channel);
            }
        }
    };

}
#endif //HAPPYML_CONCAT_WIDE_TENSOR_VIEW_H
