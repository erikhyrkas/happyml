//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP

#include "../base_tensors.hpp"
#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
    class TensorDotTensorView : public happyml::BaseTensorBinaryOperatorView {
    public:
        TensorDotTensorView(const shared_ptr<BaseTensor> &tensor1,
                            const shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->rowCount() != 1 || tensor2->rowCount() != 1 ||
                tensor1->channelCount() != 1 || tensor2->channelCount() != 1) {
                throw exception("Dot product is only applicable to 1D tensors (vectors)");
            }
            if (tensor1->columnCount() != tensor2->columnCount()) {
                throw exception("Dot product requires tensors with the same length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorDotTensorView{" << rowCount() << "," << columnCount() << "}->(";
            child1->printMaterializationPlan();
            cout << ") . (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t rowCount() override {
            return 1;
        }

        size_t columnCount() override {
            return 1;
        }

        size_t channelCount() override {
            return 1;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            float dotProduct = 0;
            size_t N = child1->columnCount();
            for (size_t i = 0; i < N; ++i) {
                dotProduct += child1->getValue(0, i, 0) * child2->getValue(0, i, 0);
            }
            return dotProduct;
        }
    };
}

#endif //HAPPYML_TENSOR_DOT_TENSOR_VIEW_HPP
