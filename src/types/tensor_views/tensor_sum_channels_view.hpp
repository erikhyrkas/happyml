//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_SUM_CHANNELS_VIEW_HPP
#define HAPPYML_TENSOR_SUM_CHANNELS_VIEW_HPP

#include "tensor_sum_to_channel_view.hpp"
#include <sstream>
#include <execution>

namespace happyml {
    class TensorSumChannelsView : public happyml::TensorSumToChannelView {
    public:
        explicit TensorSumChannelsView(const std::shared_ptr<BaseTensor> &tensor) : TensorSumToChannelView(tensor, 0, 1) {
        }

        void printMaterializationPlan() override {
            std::cout << "TensorSumChannelsView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }
    };
}
#endif //HAPPYML_TENSOR_SUM_CHANNELS_VIEW_HPP
