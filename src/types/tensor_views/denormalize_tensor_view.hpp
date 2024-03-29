//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_DENORMALIZE_TENSOR_VIEW_HPP
#define HAPPYML_DENORMALIZE_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class DenormalizeTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit DenormalizeTensorView(const std::shared_ptr<BaseTensor> &tensor, float min_val, float max_val)
                : BaseTensorUnaryOperatorView(tensor), min_val_(min_val), max_val_(max_val) {
            val_range_ = max_val_ - min_val_;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return (val * val_range_) + min_val_;
        }

        void printMaterializationPlan() override {
            cout << "TensorDenormalizeView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

    private:
        float min_val_;
        float max_val_;
        float val_range_;
    };
}
#endif //HAPPYML_DENORMALIZE_TENSOR_VIEW_HPP
