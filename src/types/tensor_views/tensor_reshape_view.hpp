//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_RESHAPE_VIEW_HPP
#define HAPPYML_TENSOR_RESHAPE_VIEW_HPP

#include <sstream>
#include <vector>
#include <execution>

namespace happyml {
// Change the number of rows and columns, but maintain the same number of elements per channel.
// You cannot change the number of channels in the current implementation.
    class TensorReshapeView : public happyml::BaseTensorUnaryOperatorView {
    public:
        TensorReshapeView(const shared_ptr<BaseTensor> &tensor, const size_t rows,
                          const size_t columns, const size_t channels) : BaseTensorUnaryOperatorView(tensor) {
            this->rows = rows;
            this->columns = columns;
            if (channels != tensor->channelCount()) {
                throw runtime_error("A matrix view must be put over a matrix with the same number of channels.");
            }
            this->elements_per_channel = (unsigned long) rows * (unsigned long) columns;
            if (tensor->elementsPerChannel() != elements_per_channel) {
                throw runtime_error("A matrix view must be put over a matrix with the same number of elements.");
            }
        }

        TensorReshapeView(const shared_ptr<BaseTensor> &tensor, const size_t rows,
                          const size_t columns) : BaseTensorUnaryOperatorView(tensor) {
            this->rows = rows;
            this->columns = columns;
            this->elements_per_channel = (unsigned long) rows * (unsigned long) columns;
            if (tensor->elementsPerChannel() != elements_per_channel) {
                throw runtime_error("A matrix view must be put over a matrix with the same number of elements.");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorReshapeView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child_->printMaterializationPlan();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return columns;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const unsigned long position_offset = ((row * columns) + column) + (channel * elements_per_channel);
            return child_->getValue(position_offset);
        }


    private:
        size_t rows;
        size_t columns;
        unsigned long elements_per_channel;
    };
}

#endif //HAPPYML_TENSOR_RESHAPE_VIEW_HPP
