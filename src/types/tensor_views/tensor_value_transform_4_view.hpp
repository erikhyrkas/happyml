//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_VALUE_TRANSFORM_4_VIEW_HPP
#define HAPPYML_TENSOR_VALUE_TRANSFORM_4_VIEW_HPP

#include <sstream>
#include <vector>
#include <utility>
#include <execution>
#include "../base_tensors.hpp"

namespace happyml {
    class TensorValueTransform4View : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorValueTransform4View(const shared_ptr<BaseTensor> &tensor,
                                  function<float(float, vector<size_t>)> transformFunction,
                                  vector<size_t> constants)
                : BaseTensorUnaryOperatorView(tensor) {
            this->transformFunction = std::move(transformFunction);
            this->constants = std::move(constants);
        }

        void printMaterializationPlan() override {
            cout << "TensorValueTransform4View{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return transformFunction(child_->getValue(row, column, channel), constants);
        }

    private:
        function<float(float, vector<size_t>)> transformFunction;
        vector<size_t> constants;
    };
}
#endif //HAPPYML_TENSOR_VALUE_TRANSFORM_4_VIEW_HPP
