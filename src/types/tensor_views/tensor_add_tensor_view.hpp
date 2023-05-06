//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_ADD_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_ADD_TENSOR_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
    class TensorAddTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorAddTensorView(const shared_ptr<BaseTensor> &tensor1,
                            const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->channelCount() != tensor2->channelCount() || tensor1->rowCount() != tensor2->rowCount() ||
                tensor1->columnCount() != tensor2->columnCount()) {
                cout << "[" << tensor1->rowCount() << ", " << tensor1->columnCount() << ", " << tensor1->channelCount()
                     << "] + [";
                cout << tensor2->rowCount() << ", " << tensor2->columnCount() << ", " << tensor2->channelCount() << "]"
                     << endl;
                throw exception("You can only add two tensors of the same dimensions together.");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorAddTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->(";
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
            return child1->getValue(row, column, channel) + child2->getValue(row, column, channel);
        }
    };
}

#endif //HAPPYML_TENSOR_ADD_TENSOR_VIEW_HPP
