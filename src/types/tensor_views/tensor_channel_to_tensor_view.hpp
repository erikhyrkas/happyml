//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_CHANNEL_TO_TENSOR_VIEW_HPP
#define HAPPYML_TENSOR_CHANNEL_TO_TENSOR_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// Creates a tensor from a single channel of another tensor, ignoring other channels
// all data is at channel 0, and channel count is 1.
    class TensorChannelToTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorChannelToTensorView(const std::shared_ptr<BaseTensor> &tensor, size_t channel_offset)
                : BaseTensorUnaryOperatorView(tensor) {
            this->channel_offset = channel_offset;
        }

        void printMaterializationPlan() override {
            cout << "TensorChannelToChannel{" << rowCount() << "," << columnCount() << ",1}->";
            child->printMaterializationPlan();
        }

        size_t channelCount() override {
            return 1;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (channel != 0) {
                return 0.f;
            }

            const float val = child->getValue(row, column, channel + channel_offset);
            return val;
        }

    private:
        size_t channel_offset;
    };
}
#endif //HAPPYML_TENSOR_CHANNEL_TO_TENSOR_VIEW_HPP
