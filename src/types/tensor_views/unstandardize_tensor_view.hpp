//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_UNSTANDARDIZE_TENSOR_VIEW_HPP
#define HAPPYML_UNSTANDARDIZE_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
    class UnstandardizeTensorView : public happyml::BaseTensorUnaryOperatorView {
    public:
        explicit UnstandardizeTensorView(const shared_ptr<BaseTensor> &tensor, float mean, float std_dev)
                : BaseTensorUnaryOperatorView(tensor), mean_(mean), std_dev_(std_dev) {
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const float val = child_->getValue(row, column, channel);
            return (val * std_dev_) + mean_;
        }

        void printMaterializationPlan() override {
            cout << "UnstandardizeTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->";
            child_->printMaterializationPlan();
        }

    private:
        float mean_;
        float std_dev_;
    };
}

#endif //HAPPYML_UNSTANDARDIZE_TENSOR_VIEW_HPP
