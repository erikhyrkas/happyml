//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_SUBTRACT_TENSOR_VIEW_HPP
#define HAPPYML_SUBTRACT_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
    // element wise subtraction
    class SubtractTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        SubtractTensorView(const shared_ptr<BaseTensor> &tensor1,
                           const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->channelCount() != tensor2->channelCount() || tensor1->rowCount() != tensor2->rowCount() ||
                tensor1->columnCount() != tensor2->columnCount()) {
                stringstream ss;
                ss << "Subtract cols, rows, and channels must match in length. Attempted: " << "[" << tensor1->rowCount()
                   << ", " << tensor1->columnCount() << ", " << tensor1->channelCount() << "] - [";
                ss << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]";
                throw runtime_error(ss.str().c_str());
            }
        }

        void printMaterializationPlan() override {
            cout << "SubtractTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") + (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return left_child_->rowCount();
        }

        size_t columnCount() override {
            return left_child_->columnCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return left_child_->getValue(row, column, channel) - right_child_->getValue(row, column, channel);
        }
    };
}

#endif //HAPPYML_SUBTRACT_TENSOR_VIEW_HPP
