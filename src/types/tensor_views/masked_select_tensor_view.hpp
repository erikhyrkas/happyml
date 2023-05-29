//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_MASKED_SELECT_TENSOR_VIEW_HPP
#define HAPPYML_MASKED_SELECT_TENSOR_VIEW_HPP

namespace happyml {
    class MaskedSelectTensorView : public BaseTensorTrinaryOperatorView {
    public:
        explicit MaskedSelectTensorView(const std::shared_ptr<BaseTensor> &mask,
                                        const std::shared_ptr<BaseTensor> &value_above_discriminator,
                                        const std::shared_ptr<BaseTensor> &value_below_discriminator,
                                        const float mask_discriminator = 0.0f)
                : BaseTensorTrinaryOperatorView(mask, value_above_discriminator, value_below_discriminator), mask_discriminator_(mask_discriminator) {}

        float getValue(size_t row, size_t column, size_t channel) override {
            const float mask_val = left_child_->getValue(row, column, channel);
            if (mask_val > mask_discriminator_) {
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
            std::cout << "MaskedSelectTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                      << "}->";
            left_child_->printMaterializationPlan();
            middle_child_->printMaterializationPlan();
            right_child_->printMaterializationPlan();
        }

    private:
        float mask_discriminator_;
    };
}
#endif //HAPPYML_MASKED_SELECT_TENSOR_VIEW_HPP
