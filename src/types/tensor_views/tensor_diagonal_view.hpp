//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_DIAGONAL_VIEW_HPP
#define HAPPYML_TENSOR_DIAGONAL_VIEW_HPP

#include <sstream>
#include <execution>
#include "../base_tensors.hpp"

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
        TensorDiagonalView(const shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
            // Assuming the input tensor is 1D or reshaped 1D
            this->n = tensor->rowCount() * tensor->columnCount();
        }

        void printMaterializationPlan() override {
            cout << "TensorDiagonalView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        size_t rowCount() override {
            return n;
        }

        size_t columnCount() override {
            return n;
        }

        bool readRowsInParallel() override {
            return false;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (row == column && row < child_->rowCount() && column < child_->columnCount() && channel < child_->channelCount()) {
                return child_->getValue(row, column, channel);
            }
            return 0.f;
        }

    private:
        size_t n;
    };
}
#endif //HAPPYML_TENSOR_DIAGONAL_VIEW_HPP
