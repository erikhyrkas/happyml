//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_EXP_VIEW_HPP
#define HAPPYML_TENSOR_EXP_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class TensorExpView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorExpView(const shared_ptr <BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return std::exp(val);
        }

        void printMaterializationPlan() override {
            cout << "TensorExpView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }
    };
}

#endif //HAPPYML_TENSOR_EXP_VIEW_HPP
