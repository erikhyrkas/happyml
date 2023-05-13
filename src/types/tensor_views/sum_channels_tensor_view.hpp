//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_SUM_CHANNELS_TENSOR_VIEW_HPP
#define HAPPYML_SUM_CHANNELS_TENSOR_VIEW_HPP

#include "sum_to_channel_tensor_view.hpp"
#include <sstream>
#include <execution>

namespace happyml {
    class SumChannelsTensorView : public SumToChannelTensorView {
    public:
        explicit SumChannelsTensorView(const std::shared_ptr<BaseTensor> &tensor) : SumToChannelTensorView(tensor, 0, 1) {
        }

        void printMaterializationPlan() override {
            std::cout << "SumChannelsTensorView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }
    };
}
#endif //HAPPYML_SUM_CHANNELS_TENSOR_VIEW_HPP
