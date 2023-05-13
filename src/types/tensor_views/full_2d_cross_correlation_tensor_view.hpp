//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FULL_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP
#define HAPPYML_FULL_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>
#include "zero_pad_tensor_view.hpp"
#include "valid_2d_cross_correlation_tensor_view.hpp"

namespace happyml {
// https://en.wikipedia.org/wiki/Cross-correlation
// https://en.wikipedia.org/wiki/Two-dimensional_correlation_analysis
// Having an even number of values in your filter is a little wierd, but I handle it
// you'll see where I do the rounding, it's so that a 2x2 filter or a 4x4 filter would work.
// I think most filters are odd numbers because even filters don't make much sense,
// but I wanted the code to work. The center of your filter is at a halfway point,
// which is why it's weird.
    class Full2DCrossCorrelationTensorView : public happyml::Valid2DCrossCorrelationTensorView {
    public:
        Full2DCrossCorrelationTensorView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : Valid2DCrossCorrelationTensorView(
                make_shared<happyml::ZeroPadTensorView>(tensor,
                                                           (kernel->rowCount() > 1) *
                                                           (size_t) std::round(((double) kernel->rowCount()) / 2.0),
                                                           (kernel->rowCount() > 1) *
                                                           (size_t) std::round(((double) kernel->rowCount()) / 2.0),
                                                           (kernel->columnCount() > 1) *
                                                           (size_t) std::round(((double) kernel->columnCount()) / 2.0),
                                                           (kernel->columnCount() > 1) *
                                                           (size_t) std::round(((double) kernel->columnCount()) / 2.0)),
                kernel) {
        }

        void printMaterializationPlan() override {
            std::cout << "TensorFullCrossCorrelation2dView{" << rowCount() << "," << columnCount() << "," << channelCount()
                      << "}->(";
            left_child_->printMaterializationPlan();
            std::cout << ") + (";
            right_child_->printMaterializationPlan();
            std::cout << ")";
        }

    private:
    };
}

#endif //HAPPYML_FULL_2D_CROSS_CORRELATION_TENSOR_VIEW_HPP
