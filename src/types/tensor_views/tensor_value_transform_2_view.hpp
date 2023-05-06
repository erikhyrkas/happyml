//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP
#define HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP

#include <sstream>
#include <vector>
#include <utility>
#include <execution>

namespace happyml {
    class TensorValueTransform2View : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorValueTransform2View(const shared_ptr<BaseTensor> &tensor,
                                  function<float(float, vector<double>)> transformFunction,
                                  vector<double> constants) : BaseTensorUnaryOperatorView(
                tensor) {
            this->transformFunction = std::move(transformFunction);
            this->constants = std::move(constants);
        }

        void printMaterializationPlan() override {
            cout << "TensorValueTransform2View{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return transformFunction(child->getValue(row, column, channel), constants);
        }

    private:
        function<float(float, vector<double>)> transformFunction;
        vector<double> constants;
    };
}
#endif //HAPPYML_TENSOR_VALUE_TRANSFORM_2_VIEW_HPP
