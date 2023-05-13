//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP
#define HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP

#include <sstream>
#include <vector>
#include <utility>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    class ValueTransformWithDoubleTensorView : public BaseTensorUnaryOperatorView {
    public:
        ValueTransformWithDoubleTensorView(const shared_ptr<BaseTensor> &tensor,
                                  function<float(float, double)> transformFunction,
                                  double constant)
                : BaseTensorUnaryOperatorView(tensor) {
            this->transformFunction = std::move(transformFunction);
            this->constant_ = constant;
        }

        void printMaterializationPlan() override {
            cout << "ValueTransformWithDoubleTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return transformFunction(child_->getValue(row, column, channel), constant_);
        }

    private:
        function<float(float, double)> transformFunction;
        double constant_;
    };
}
#endif //HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP
