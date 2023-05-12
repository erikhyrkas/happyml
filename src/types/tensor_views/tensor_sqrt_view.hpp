//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_SQRT_VIEW_HPP
#define HAPPYML_TENSOR_SQRT_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {

    class TensorSqrtView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit TensorSqrtView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(
                tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return sqrt(val);
        }

        void printMaterializationPlan() override {
            cout << "TensorSqrtView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    };
}

#endif //HAPPYML_TENSOR_SQRT_VIEW_HPP
