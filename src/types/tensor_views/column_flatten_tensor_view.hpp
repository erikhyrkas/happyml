//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_FLATTEN_TO_COLUMN_VIEW_HPP
#define HAPPYML_TENSOR_FLATTEN_TO_COLUMN_VIEW_HPP

#include "../base_tensors.hpp"
#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
// Converts a 3d tensor into a column vector
    class ColumnFlattenTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit ColumnFlattenTensorView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(
                tensor) {
            this->rows = tensor->size();
        }

        void printMaterializationPlan() override {
            cout << "ColumnFlattenTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return 1;
        }

        size_t channelCount() override {
            return 1;
        }

        bool readRowsInParallel() override {
            return true;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (column != 0 || channel != 0) {
                throw runtime_error("Column Vector has only a single column and channel.");
            }
            return child_->getValue(row);
        }


    private:
        size_t rows;
    };
}
#endif //HAPPYML_TENSOR_FLATTEN_TO_COLUMN_VIEW_HPP
