//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_TRANSPOSE_VIEW_HPP
#define HAPPYML_TENSOR_TRANSPOSE_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorTransposeView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit TensorTransposeView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "TensorTransposeView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        size_t rowCount() override {
            return child->columnCount();
        }

        size_t columnCount() override {
            return child->rowCount();
        }

        size_t channelCount() override {
            return child->channelCount();
        }

        bool readRowsInParallel() override {
            return !child->readRowsInParallel();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            // making it obvious that we intend to swap column and row. Compiler will optimize this out.
            const size_t swapped_row = column;
            const size_t swapped_col = row;
            return child->getValue(swapped_row, swapped_col, channel);
        }
    };
}

#endif //HAPPYML_TENSOR_TRANSPOSE_VIEW_HPP
