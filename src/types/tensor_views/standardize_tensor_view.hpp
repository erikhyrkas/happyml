//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_STANDARDIZE_TENSOR_VIEW_HPP
#define HAPPYML_STANDARDIZE_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class StandardizeTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit StandardizeTensorView(const shared_ptr<BaseTensor> &tensor)
                : BaseTensorUnaryOperatorView(tensor) {
            double mean = 0.0;
            double M2 = 0.0;
            double total_elements = tensor->size();
            for (size_t element = 0; element < total_elements; ++element) {
                double value = tensor->getValue(element);
                double delta = value - mean;
                mean += delta / total_elements;
                double delta2 = value - mean;
                M2 += delta * delta2;
            }
            double variance = M2 / total_elements;
            mean_ = (float) mean;
            if(variance == 0) {
                std_dev_ = 1.0f;
            } else {
                std_dev_ = (float) sqrt(variance);
            }
        }

        explicit StandardizeTensorView(const shared_ptr<BaseTensor> &tensor, float mean, float std_dev)
                : BaseTensorUnaryOperatorView(tensor), mean_(mean), std_dev_(std_dev) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return (val - mean_) / std_dev_;
        }

        void printMaterializationPlan() override {
            cout << "StandardizeTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

        float get_mean() const {
            return mean_;
        }

        float get_std_dev() const {
            return std_dev_;
        }
    private:
        float mean_;
        float std_dev_;
    };
}

#endif //HAPPYML_STANDARDIZE_TENSOR_VIEW_HPP
