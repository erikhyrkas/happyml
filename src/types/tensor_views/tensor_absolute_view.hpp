//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_ABSOLUTE_VIEW_HPP
#define HAPPYML_TENSOR_ABSOLUTE_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    // absolute value of each value
    class TensorAbsoluteView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorAbsoluteView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return std::abs(val);
        }

        void printMaterializationPlan() override {
            std::cout << "TensorAbsoluteView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }
    };
}

#endif //HAPPYML_TENSOR_ABSOLUTE_VIEW_HPP
