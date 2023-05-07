//
// Created by Erik Hyrkas on 5/6/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_LESS_THAN_SCALAR_VIEW_HPP
#define HAPPYML_TENSOR_LESS_THAN_SCALAR_VIEW_HPP

namespace happyml {
    class TensorLessThanScalarView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorLessThanScalarView(const std::shared_ptr <BaseTensor> &tensor, float scalar)
                : BaseTensorUnaryOperatorView(tensor), scalar_(scalar) {}

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return val < scalar_ ? 1.0f : 0.0f;
        }

        void printMaterializationPlan() override {
            std::cout << "TensorLessThanScalarView{" << rowCount() << "," << columnCount() << "," << channelCount()
                      << "}->";
            child->printMaterializationPlan();
        }

    private:
        float scalar_;
    };
}
#endif //HAPPYML_TENSOR_LESS_THAN_SCALAR_VIEW_HPP
