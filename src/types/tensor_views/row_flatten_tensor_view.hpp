//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_ROW_FLATTEN_TENSOR_VIEW_HPP
#define HAPPYML_ROW_FLATTEN_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
// Converts a 3d tensor into a row vector
    class RowFlattenTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit RowFlattenTensorView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
            this->columns = tensor->size();
        }

        void printMaterializationPlan() override {
            cout << "RowFlattenTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        size_t rowCount() override {
            return 1;
        }

        size_t columnCount() override {
            return columns;
        }

        size_t channelCount() override {
            return 1;
        }

        bool readRowsInParallel() override {
            return false;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (row != 0 || channel != 0) {
                throw runtime_error("Row Vector has only a single row and channel.");
            }
            return child_->getValue(column);
        }


    private:
        size_t columns;
    };
}
#endif //HAPPYML_ROW_FLATTEN_TENSOR_VIEW_HPP
