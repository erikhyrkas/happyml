//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FULL_2D_CONVOLVE_TENSOR_VIEW_HPP
#define HAPPYML_FULL_2D_CONVOLVE_TENSOR_VIEW_HPP

#include <execution>
#include <sstream>
#include "rotate_180_tensor_view.hpp"
#include "full_2d_cross_correlation_tensor_view.hpp"


using namespace std;

namespace happyml {
    // A convolved tensor appears to be equivalent to cross correlation with the filter rotated 180 degrees:
    // https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class Full2DConvolveTensorView : public Full2DCrossCorrelationTensorView {
    public:
        Full2DConvolveTensorView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : Full2DCrossCorrelationTensorView(tensor, make_shared<Rotate180TensorView>(kernel)) {

        }

        void printMaterializationPlan() override {
            cout << "Full2DConvolveTensorView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") + (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }
    };
}
#endif //HAPPYML_FULL_2D_CONVOLVE_TENSOR_VIEW_HPP
