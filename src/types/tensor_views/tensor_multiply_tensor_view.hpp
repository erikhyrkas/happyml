//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_MULTIPLY_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_MULTIPLY_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
// element-wise multiplication (Hadamard product/entry-wise product)
    class TensorMultiplyTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorMultiplyTensorView(const shared_ptr<BaseTensor> &tensor1,
                                 const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1,
                                                                                                       tensor2) {
            if (tensor1->columnCount() != tensor2->columnCount() || tensor1->rowCount() != tensor2->rowCount()) {
                stringstream ss;
                ss << "Multiply cols and rows much match in length. Attempted: " << "[" << tensor1->rowCount() << ", "
                   << tensor1->columnCount() << ", " << tensor1->channelCount() << "] * [";
                ss << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                   << endl;
                throw exception(ss.str().c_str());
            }
            if (tensor1->channelCount() != tensor2->channelCount()) {
                stringstream ss;
                ss << "Multiply product channels must match in length. Attempted: " << "[" << tensor1->rowCount()
                   << ", " << tensor1->columnCount() << ", " << tensor1->channelCount() << "] * [";
                ss << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                   << endl;
                throw exception(ss.str().c_str());

            }
        }

        void printMaterializationPlan() override {
            cout << "TensorMultiplyTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
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
            return child1->columnCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
//        cout << "getting val: " << row << ", " << column << endl;
            return child1->getValue(row, column, channel) * child2->getValue(row, column, channel);
        }
    };
}

#endif //HAPPYML_TENSOR_MULTIPLY_TENSOR_VIEW_HPP
