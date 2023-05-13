//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_VALUE_TRANSFORM_TENSOR_VIEW_HPP
#define HAPPYML_VALUE_TRANSFORM_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class ValueTransformTensorView : public BaseTensorUnaryOperatorView {
    public:
        ValueTransformTensorView(const shared_ptr<BaseTensor> &tensor, function<float(float)> transformFunction)
                : BaseTensorUnaryOperatorView(
                tensor) {
            this->transformFunction = std::move(transformFunction);
        }

        void printMaterializationPlan() override {
            cout << "ValueTransformTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return transformFunction(child_->getValue(row, column, channel));
        }

    private:
        function<float(float)> transformFunction;
    };
}

#endif //HAPPYML_VALUE_TRANSFORM_TENSOR_VIEW_HPP
