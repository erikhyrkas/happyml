//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_POWER_VIEW_HPP
#define HAPPYML_TENSOR_POWER_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>

namespace happyml {
    class TensorPowerView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorPowerView(const shared_ptr<BaseTensor> &tensor, const float power) : BaseTensorUnaryOperatorView(
                tensor) {
            this->power = power;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return powf(val, power);
        }

        void printMaterializationPlan() override {
            cout << "TensorPowerView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

    private:
        float power;
    };
}

#endif //HAPPYML_TENSOR_POWER_VIEW_HPP
