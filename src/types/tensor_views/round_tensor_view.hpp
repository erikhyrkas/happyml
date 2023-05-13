//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_ROUND_TENSOR_VIEW_HPP
#define HAPPYML_ROUND_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class RoundTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit RoundTensorView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "RoundTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return round(val);
        }

    private:
    };
}

#endif //HAPPYML_ROUND_TENSOR_VIEW_HPP
