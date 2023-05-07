//
// Created by Erik Hyrkas on 5/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOROUTERPRODUCTVIEW_HPP
#define HAPPYML_TENSOROUTERPRODUCTVIEW_HPP

#include "../base_tensors.hpp"

namespace happyml {
    class TensorOuterProductView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorOuterProductView(const shared_ptr<BaseTensor> &left, const shared_ptr<BaseTensor> &right)
                : BaseTensorBinaryOperatorView(left, right) {}

        void printMaterializationPlan() override {
            cout << "TensorOuterProductView{" << rowCount() << "," << columnCount() << "," << channelCount()
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
            return left_child_->getValue(row, 0, channel) * right_child_->getValue(0, column, channel);
        }
    };

}
#endif //HAPPYML_TENSOROUTERPRODUCTVIEW_HPP
