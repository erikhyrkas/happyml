//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP

#include "../base_tensors.hpp"
#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
    class TensorDotTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorDotTensorView(const shared_ptr<BaseTensor> &tensor1,
                            const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->rowCount() != 1 || tensor2->rowCount() != 1 ||
                tensor1->channelCount() != 1 || tensor2->channelCount() != 1) {
                throw runtime_error("Dot product is only applicable to 1D tensors (vectors)");
            }
            if (tensor1->columnCount() != tensor2->columnCount()) {
                throw runtime_error("Dot product requires tensors with the same length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorDotTensorView{" << rowCount() << "," << columnCount() << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") . (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return 1;
        }

        size_t columnCount() override {
            return 1;
        }

        size_t channelCount() override {
            return 1;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            float dotProduct = 0;
            size_t N = left_child_->columnCount();
            for (size_t i = 0; i < N; ++i) {
                dotProduct += left_child_->getValue(0, i, 0) * right_child_->getValue(0, i, 0);
            }
            return dotProduct;
        }
    };
}

#endif //HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP
