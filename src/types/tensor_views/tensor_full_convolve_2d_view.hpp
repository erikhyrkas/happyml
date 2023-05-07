//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_FULL_CONVOLVE_2D_VIEW_HPP
#define HAPPYML_TENSOR_FULL_CONVOLVE_2D_VIEW_HPP

#include <execution>
#include <sstream>
#include "tensor_rotate_180_view.hpp"
#include "tensor_full_cross_correlation_2d_view.hpp"


using namespace std;

namespace happyml {
    // A convolved tensor appears to be equivalent to cross correlation with the filter rotated 180 degrees:
    // https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class TensorFullConvolve2dView : public TensorFullCrossCorrelation2dView {
    public:
        TensorFullConvolve2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : TensorFullCrossCorrelation2dView(tensor, make_shared<TensorRotate180View>(kernel)) {

        }

        void printMaterializationPlan() override {
            cout << "TensorFullConvolve2dView{" << rowCount() << "," << columnCount() << "," << channelCount()
                 << "}->(";
            left_child_->printMaterializationPlan();
            cout << ") + (";
            right_child_->printMaterializationPlan();
            cout << ")";
        }
    };
}
#endif //HAPPYML_TENSOR_FULL_CONVOLVE_2D_VIEW_HPP
