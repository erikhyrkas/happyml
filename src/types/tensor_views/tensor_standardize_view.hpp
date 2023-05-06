//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_STANDARDIZE_VIEW_HPP
#define HAPPYML_TENSOR_STANDARDIZE_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class TensorStandardizeView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit TensorStandardizeView(const shared_ptr<BaseTensor> &tensor, float mean, float std_dev)
                : BaseTensorUnaryOperatorView(tensor), mean_(mean), std_dev_(std_dev) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child->getValue(row, column, channel);
            return (val - mean_) / std_dev_;
        }

        void printMaterializationPlan() override {
            cout << "TensorStandardizeView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child->printMaterializationPlan();
        }

    private:
        float mean_;
        float std_dev_;
    };
}

#endif //HAPPYML_TENSOR_STANDARDIZE_VIEW_HPP
