//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_SQRT_TENSOR_VIEW_HPP
#define HAPPYML_SQRT_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {

    class SqrtTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit SqrtTensorView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(
                tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return sqrt(val);
        }

        void printMaterializationPlan() override {
            cout << "SqrtTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    };
}

#endif //HAPPYML_SQRT_TENSOR_VIEW_HPP
