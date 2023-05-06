//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
// matmul
    class TensorMatrixMultiplyTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorMatrixMultiplyTensorView(const shared_ptr<BaseTensor> &tensor1,
                                       const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1,
                                                                                                             tensor2) {
            if (tensor1->columnCount() != tensor2->rowCount()) {
                cout << "[" << tensor1->rowCount() << ", " << tensor1->columnCount() << ", " << tensor1->channelCount()
                     << "] dot [";
                cout << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                     << endl;
                throw exception("Dot product tensor1.cols must match tensor2.rows in length");
            }
            if (tensor1->channelCount() != tensor2->channelCount()) {
                throw exception("Dot product tensor1.channels must match tensor2.channels in length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorMatrixMultiplyTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return child1->rowCount();
        }

        size_t columnCount() override {
            return child2->columnCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            float val = 0;
            const auto childColumnCount = child1->columnCount();
//#pragma omp for
            for (long long t1_col = 0; t1_col < childColumnCount; t1_col++) {
                val += child1->getValue(row, t1_col, channel) * child2->getValue(t1_col, column, channel);
            }
            return val;
        }
    };
}

#endif //HAPPYML_TENSOR_MATRIX_MULTIPLY_TENSOR_VIEW_HPP
