//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_NOOP_VIEW_HPP
#define HAPPYML_TENSOR_NOOP_VIEW_HPP

#include <sstream>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    class TensorNoOpView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorNoOpView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {}

        float getValue(size_t row, size_t column, size_t channel) override {
            return child_->getValue(row, column, channel);
        }

        void printMaterializationPlan() override {
            std::cout << "NoOpTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    private:
    };
}

#endif //HAPPYML_TENSOR_NOOP_VIEW_HPP
