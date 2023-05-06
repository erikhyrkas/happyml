//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_LOG2_VIEW_HPP
#define HAPPYML_TENSOR_LOG2_VIEW_HPP

#include <sstream>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    class TensorLog2View : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorLog2View(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "TensorLog2View{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return log2(val);
        }

    private:
    };
}
#endif //HAPPYML_TENSOR_LOG2_VIEW_HPP
