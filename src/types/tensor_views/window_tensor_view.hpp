//
// Created by Erik Hyrkas on 5/11/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSORWINDOWVIEW_H
#define HAPPYML_TENSORWINDOWVIEW_H

#include "../base_tensors.hpp"

namespace happyml {

    class WindowTensorView : public happyml::BaseTensorUnaryOperatorView {
    public:
        WindowTensorView(const shared_ptr<BaseTensor> &tensor, size_t start_column, size_t end_column)
                : BaseTensorUnaryOperatorView(tensor), start_column_(start_column), end_column_(end_column) {
            if (start_column >= end_column || end_column > tensor->columnCount()) {
                throw runtime_error("Invalid window range for TensorWindowView.");
            }
        }

        void printMaterializationPlan() override {
            cout << "WindowTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        size_t columnCount() override {
            return end_column_ - start_column_;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (column >= columnCount()) {
                throw runtime_error("Column index out of range for TensorWindowView.");
            }
            return child_->getValue(row, column + start_column_, channel);
        }

    private:
        size_t start_column_;
        size_t end_column_;
    };

}
#endif //HAPPYML_TENSORWINDOWVIEW_H
