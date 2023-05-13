//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_CLIP_TENSOR_VIEW_HPP
#define HAPPYML_CLIP_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
// Prevents the tensor's value from being bigger than or less than a value
    class ClipTensorView : public BaseTensorUnaryOperatorView {
    public:
        ClipTensorView(const std::shared_ptr<BaseTensor> &tensor, float min_value, float max_value) : BaseTensorUnaryOperatorView(
                tensor) {
            this->max_value = max_value;
            this->min_value = min_value;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return std::clamp(val, min_value, max_value);
        }

        void printMaterializationPlan() override {
            cout << "ClipTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    private:
        float max_value;
        float min_value;
    };
}
#endif //HAPPYML_CLIP_TENSOR_VIEW_HPP
