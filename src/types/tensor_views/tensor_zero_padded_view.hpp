//
// Created by Erik Hyrkas on 5/6/2023.
//

#ifndef HAPPYML_TENSOR_ZERO_PADDED_VIEW_HPP
#define HAPPYML_TENSOR_ZERO_PADDED_VIEW_HPP

#include <sstream>
#include <execution>

namespace happyml {
// padding is the amount of extra cells on a given "side" of the matrix
// so a col_padding of 2 would mean 2 cells to the left that are 0 and 2 cells to the right that are zero.
// for a total of 4 extra cells in the row.
    class TensorZeroPaddedView : public BaseTensorUnaryOperatorView {
    public:
        TensorZeroPaddedView(const std::shared_ptr<BaseTensor> &tensor, size_t topPadding, size_t bottomPadding,
                             size_t leftPadding, size_t rightPadding) : BaseTensorUnaryOperatorView(tensor) {
            this->topPadding = topPadding;
            this->bottomPadding = bottomPadding;
            this->leftPadding = leftPadding;
            this->rightPadding = rightPadding;
        }

        void printMaterializationPlan() override {
            cout << "TensorZeroPaddedView{" << rowCount() << "," << columnCount() << "," << channelCount() << "}->";
            child->printMaterializationPlan();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            const auto adjusted_row = row - topPadding;
            const auto adjusted_col = column - leftPadding;
            // todo: for performance, we can potentially store rowcount as a constant when we make this object.
            if (row < topPadding || adjusted_row >= child->rowCount() ||
                column < leftPadding || adjusted_col >= child->columnCount()) {
                return 0.f;
            }
            const float val = child->getValue(adjusted_row, adjusted_col, channel);
            return val;
        }

        size_t rowCount() override {
            const size_t padding = bottomPadding + topPadding;
            return child->rowCount() + padding;
        }

        size_t columnCount() override {
            const size_t padding = rightPadding + leftPadding;
            return child->columnCount() + padding;
        }

    private:
        size_t topPadding;
        size_t bottomPadding;
        size_t leftPadding;
        size_t rightPadding;
    };
}
#endif //HAPPYML_TENSOR_ZERO_PADDED_VIEW_HPP
