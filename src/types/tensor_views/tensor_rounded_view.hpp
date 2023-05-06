//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_ROUNDED_VIEW_HPP
#define HAPPYML_TENSOR_ROUNDED_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorRoundedView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorRoundedView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "TensorRoundedView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return round(val);
        }

    private:
    };
}

#endif //HAPPYML_TENSOR_ROUNDED_VIEW_HPP
