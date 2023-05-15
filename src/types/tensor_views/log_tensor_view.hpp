//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_LOG_TENSOR_VIEW_HPP
#define HAPPYML_LOG_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    class LogTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit LogTensorView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "LogTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return std::log(std::clamp(val, 1e-8f, 1.0f - 1e-8f));
        }
    };
}

#endif //HAPPYML_LOG_TENSOR_VIEW_HPP
