//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_DIAGONAL_VIEW_HPP
#define HAPPYML_TENSOR_DIAGONAL_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// In the current implementation, a tensor is a vector of matrices, and our math is frequently
// interested in each matrix rather than treating the tensor as a whole, so this implementation
// returns the diagonal of each matrix in the tensor.
//
// 0, 1, 2
// 3, 4, 5   becomes  0, 4, 8
// 6, 7, 8
//
// If the tensor has more channels, we do the same thing for each channel.
// If you want to learn more about eiganvalues and diagonalization, and you don't mind
// a lot of math jargon, read here:
// https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
// or here:
// https://mathworld.wolfram.com/MatrixDiagonalization.html
//
// Personally, my linear algebra class was ~25 years ago, and I found this refresher
// useful: https://www.youtube.com/playlist?list=PLybg94GvOJ9En46TNCXL2n6SiqRc_iMB8
// and specifically: https://www.youtube.com/watch?v=WTLl03D4TNA
    class TensorDiagonalView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorDiagonalView(const shared_ptr<BaseTensor> &tensor, size_t row_offset) : BaseTensorUnaryOperatorView(
                tensor) {
            this->row_offset = row_offset;
            this->is_1d = tensor->rowCount() == 1;
            if (!is_1d) {
                // we only have as many columns as there were rows
                this->columns = tensor->rowCount() - row_offset;
                // we either have 0 or 1 result row
                this->rows = row_offset < tensor->rowCount();
            } else {
                this->columns = tensor->columnCount() - row_offset;
                this->rows = this->columns;
            }

        }

        void printMaterializationPlan() override {
            cout << "TensorDiagonalView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        explicit TensorDiagonalView(const shared_ptr<BaseTensor> &tensor)
                : TensorDiagonalView(tensor, 0) {
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return columns;
        }

        bool readRowsInParallel() override {
            return false;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (is_1d) {
                if (row + row_offset == column) {
                    child->getValue(0, column, channel);
                }
                return 0.f;
            }
            // we aren't bounds checking, so the caller better make sure that row_count > 0
            return child->getValue(column + row_offset, column, channel);
        }

    private:
        size_t row_offset;
        size_t columns;
        size_t rows;
        bool is_1d;
    };
}
#endif //HAPPYML_TENSOR_DIAGONAL_VIEW_HPP
