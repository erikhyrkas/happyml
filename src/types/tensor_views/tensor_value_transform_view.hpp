//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_VALUE_TRANSFORM_VIEW_HPP
#define HAPPYML_TENSOR_VALUE_TRANSFORM_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class TensorValueTransformView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorValueTransformView(const shared_ptr<BaseTensor> &tensor, function<float(float)> transformFunction)
                : BaseTensorUnaryOperatorView(
                tensor) {
            this->transformFunction = std::move(transformFunction);
        }

        void printMaterializationPlan() override {
            cout << "TensorValueTransformView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return transformFunction(child_->getValue(row, column, channel));
        }

    private:
        function<float(float)> transformFunction;
    };
}

#endif //HAPPYML_TENSOR_VALUE_TRANSFORM_VIEW_HPP
