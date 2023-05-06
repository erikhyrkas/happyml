//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_FULL_CROSS_CORRELATION_2D_VIEW_HPP
#define HAPPYML_TENSOR_FULL_CROSS_CORRELATION_2D_VIEW_HPP

#include <sstream>
#include <utility>
#include <execution>
#include "tensor_zero_padded_view.hpp"
#include "tensor_valid_cross_correlation_2d_view.hpp"

namespace happyml {
// https://en.wikipedia.org/wiki/Cross-correlation
// https://en.wikipedia.org/wiki/Two-dimensional_correlation_analysis
// Having an even number of values in your filter is a little wierd, but I handle it
// you'll see where I do the rounding, it's so that a 2x2 filter or a 4x4 filter would work.
// I think most filters are odd numbers because even filters don't make much sense,
// but I wanted the code to work. The center of your filter is at a halfway point,
// which is why it's weird.
    class TensorFullCrossCorrelation2dView : public happyml::TensorValidCrossCorrelation2dView {
    public:
        TensorFullCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : TensorValidCrossCorrelation2dView(
                make_shared<happyml::TensorZeroPaddedView>(tensor,
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
            child1->printMaterializationPlan();
            std::cout << ") + (";
            child2->printMaterializationPlan();
            std::cout << ")";
        }

    private:
    };
}

#endif //HAPPYML_TENSOR_FULL_CROSS_CORRELATION_2D_VIEW_HPP
