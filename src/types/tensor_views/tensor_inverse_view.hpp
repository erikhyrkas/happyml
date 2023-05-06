//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_INVERSE_VIEW_HPP
#define HAPPYML_TENSOR_INVERSE_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// returns inverse of each value( 1.0f / original value )
    class TensorInverseView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorInverseView(const std::shared_ptr<BaseTensor> &tensor, float epsilon = 1e-8) : BaseTensorUnaryOperatorView(
                tensor) {
            this->epsilon = epsilon;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel) + epsilon;
            return 1.0f / val;
        }

        void printMaterializationPlan() override {
            cout << "TensorInverseView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

    private:
        float epsilon;
    };
}

#endif //HAPPYML_TENSOR_INVERSE_VIEW_HPP
