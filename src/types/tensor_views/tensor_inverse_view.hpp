//
// Created by Erik Hyrkas on 5/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_INVERT_VIEW_HPP
#define HAPPYML_TENSOR_INVERT_VIEW_HPP

namespace happyml {
    // An inverse function that completes the Gauss-Jordan elimination method to compute the inverse of a given matrix
    // Used in the TensorMatrixDivideTensorView class

    // Helper function to compute the inverse of a matrix using the Gauss-Jordan elimination method
    shared_ptr<BaseTensor> inverse_tensor(const shared_ptr<BaseTensor> &tensor) {
        vector<vector<vector<float>>> result;
        for (size_t channel = 0; channel < tensor->channelCount(); channel++) {
            // Create a matrix from the tensor (only one channel
            size_t n = tensor->rowCount();
            vector<vector<float>> mat(n, vector<float>(n * 2));

            // Create an augmented matrix [A|I]
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    mat[i][j] = tensor->getValue(i, j, channel);
                    mat[i][j + n] = (i == j) ? 1 : 0;
                }
            }

            // Perform Gauss-Jordan elimination
            for (size_t i = 0; i < n; i++) {
                if (mat[i][i] == 0) {
                    size_t row_to_swap = i;
                    for (size_t j = i + 1; j < n; j++) {
                        if (mat[j][i] != 0) {
                            row_to_swap = j;
                            break;
                        }
                    }
                    if (row_to_swap == i) {
                        throw runtime_error("Matrix is not invertible");
                    }
                    swap(mat[i], mat[row_to_swap]);
                }

                float pivot = mat[i][i];
                for (size_t j = i; j < 2 * n; j++) {
                    mat[i][j] /= pivot;
                }
                for (size_t j = 0; j < n; j++) {
                    if (j != i) {
                        float factor = mat[j][i];
                        for (size_t k = 0; k < 2 * n; k++) {
                            mat[j][k] -= factor * mat[i][k];
                        }
                    }
                }
            }

            // Extract the inverted matrix
            vector<vector<float>> inv(n, vector<float>(n));
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    inv[i][j] = mat[i][j + n];
                }
            }
            result.push_back(inv);
        }
        return make_shared<FullTensor>(result);
    }

    class TensorInverseView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorInverseView(const std::shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(inverse_tensor(tensor)) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return child->getValue(row, column, channel);
        }

        void printMaterializationPlan() override {
            cout << "TensorInverseView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }
    };
}

#endif //HAPPYML_TENSOR_INVERT_VIEW_HPP
