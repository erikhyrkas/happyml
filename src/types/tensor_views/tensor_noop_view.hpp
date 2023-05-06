//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_NOOP_VIEW_HPP
#define HAPPYML_TENSOR_NOOP_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorNoOpView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit TensorNoOpView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {}

        float getValue(size_t row, size_t column, size_t channel) override {
            return child->getValue(row, column, channel);
        }

        void printMaterializationPlan() override {
            cout << "TensorNoOpView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

    private:
    };
}

#endif //HAPPYML_TENSOR_NOOP_VIEW_HPP
