//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_CLIP_VIEW_HPP
#define HAPPYML_TENSOR_CLIP_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
// Prevents the tensor's value from being bigger than or less than a value
    class TensorClipView : public BaseTensorUnaryOperatorView {
    public:
        TensorClipView(const std::shared_ptr<BaseTensor> &tensor, float min_value, float max_value) : BaseTensorUnaryOperatorView(
                tensor) {
            this->max_value = max_value;
            this->min_value = min_value;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return std::clamp(val, min_value, max_value);
        }

        void printMaterializationPlan() override {
            cout << "TensorClipView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

    private:
        float max_value;
        float min_value;
    };
}
#endif //HAPPYML_TENSOR_CLIP_VIEW_HPP
