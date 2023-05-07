//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_MASKED_SELECT_VIEW_HPP
#define HAPPYML_TENSOR_MASKED_SELECT_VIEW_HPP

namespace happyml {
    class TensorMaskedSelectView : public BaseTensorTrinaryOperatorView {
    public:
        explicit TensorMaskedSelectView(const std::shared_ptr<BaseTensor> &mask,
                                        const std::shared_ptr<BaseTensor> &tensor1,
                                        const std::shared_ptr<BaseTensor> &tensor2)
                : BaseTensorTrinaryOperatorView(mask, tensor1, tensor2) {}

        float getValue(size_t row, size_t column, size_t channel) override {
            const float mask_val = left_child_->getValue(row, column, channel);
            if (mask_val > 0.0f) {
                return middle_child_->getValue(row, column, channel);
            } else {
                return right_child_->getValue(row, column, channel);
            }
        }

        size_t rowCount() override {
            return left_child_->rowCount();
        }

        size_t columnCount() override {
            return left_child_->columnCount();
        }

        size_t channelCount() override {
            return left_child_->channelCount();
        }

        void printMaterializationPlan() override {
            std::cout << "TensorMaskedSelectView{" << rowCount() << "," << columnCount() << "," << channelCount()
                      << "}->";
            left_child_->printMaterializationPlan();
            middle_child_->printMaterializationPlan();
            right_child_->printMaterializationPlan();
        }
    };
}
#endif //HAPPYML_TENSOR_MASKED_SELECT_VIEW_HPP
