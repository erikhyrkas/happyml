//
// Created by Erik Hyrkas on 5/31/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_SOFTMAX_DERIVATIVE_TENSOR_VIEW_HPP
#define HAPPYML_SOFTMAX_DERIVATIVE_TENSOR_VIEW_HPP

// This isn't used right now, but I've implemented it like 12 times as I ponder if there is any better
// way to do softmax.
namespace happyml {
    class SoftmaxDerivativeTensorView : public BaseTensorBinaryOperatorView {
    public:
        SoftmaxDerivativeTensorView(const shared_ptr<BaseTensor> &softmax_of_prediction, const shared_ptr<BaseTensor> &truth)
                : BaseTensorBinaryOperatorView(softmax_of_prediction, truth) {
        }

        void printMaterializationPlan() override {
            cout << "SoftmaxDerivativeTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            left_child_->printMaterializationPlan();
            right_child_->printMaterializationPlan();
        }

        size_t rowCount() override {
            return left_child_->rowCount();
        }

        size_t columnCount() override {
            return left_child_->columnCount();
        }

        size_t channelCount() override {
            return left_child_->channelCount();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            // Kronecker delta is 1 if i == j, 0 otherwise
            // softmax derivative is kronecker delta - softmax
            float softmax_of_prediction = left_child_->getValue(row, column, channel);
            auto truth_index = right_child_->maxIndexByRow(channel, row); // column that is 1.0f
            float kronecker_delta = (column == truth_index) ? 1.0f : 0.0f;
            return kronecker_delta - softmax_of_prediction;
        }
    };
}
#endif //HAPPYML_SOFTMAX_DERIVATIVE_TENSOR_VIEW_HPP
