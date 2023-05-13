//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_ABSOLUTE_TENSOR_VIEW_HPP
#define HAPPYML_ABSOLUTE_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    // absolute value of each value
    class AbsoluteTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit AbsoluteTensorView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return std::abs(val);
        }

        void printMaterializationPlan() override {
            std::cout << "AbsoluteTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }
    };
}

#endif //HAPPYML_ABSOLUTE_TENSOR_VIEW_HPP
