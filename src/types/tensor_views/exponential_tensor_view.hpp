//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_EXPONENTIAL_TENSOR_VIEW_HPP
#define HAPPYML_EXPONENTIAL_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class ExponentialTensorView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit ExponentialTensorView(const shared_ptr <BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return std::exp(val);
        }

        void printMaterializationPlan() override {
            cout << "ExponentialTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }
    };
}

#endif //HAPPYML_EXPONENTIAL_TENSOR_VIEW_HPP
