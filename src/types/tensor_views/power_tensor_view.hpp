//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_POWER_TENSOR_VIEW_HPP
#define HAPPYML_POWER_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class PowerTensorView : public BaseTensorUnaryOperatorView {
    public:
        PowerTensorView(const shared_ptr<BaseTensor> &tensor, const float power) : BaseTensorUnaryOperatorView(
                tensor) {
            this->power = power;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return powf(val, power);
        }

        void printMaterializationPlan() override {
            cout << "PowerTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

    private:
        float power;
    };
}

#endif //HAPPYML_POWER_TENSOR_VIEW_HPP
