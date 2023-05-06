//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_SUM_TO_CHANNEL_VIEW_HPP
#define HAPPYML_TENSOR_SUM_TO_CHANNEL_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// For a given tensor, sum the all values at a column and row and place at a specific channel index, while other channels
// are all zero. This allows us to not only sum the tensors channels into a single channel,
// but combine the resulting tensor with other tensors.
    class TensorSumToChannelView : public BaseTensorUnaryOperatorView {
    public:
        TensorSumToChannelView(const std::shared_ptr<BaseTensor> &tensor, size_t data_channel_index,
                               size_t number_of_channels) : BaseTensorUnaryOperatorView(tensor) {
            this->data_channel_index = data_channel_index;
            this->number_of_channels = number_of_channels;
        }

        void printMaterializationPlan() override {
            cout << "TensorSumToChannelView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        size_t channelCount() override {
            return number_of_channels;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            if (channel != data_channel_index) {
                return 0.f;
            }
            float result = 0.f;
            const size_t channels = child->channelCount();
//#pragma omp for
            for (long long next_channel = 0; next_channel < channels; next_channel++) {
                result += child->getValue(row, column, next_channel);
            }
            return result;
        }

    private:
        size_t data_channel_index;
        size_t number_of_channels;
    };
}

#endif //HAPPYML_TENSOR_SUM_TO_CHANNEL_VIEW_HPP
