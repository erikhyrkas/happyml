//
// Created by Erik Hyrkas on 12/9/2022.
//

#ifndef MICROML_TENSOR_VIEWS_HPP
#define MICROML_TENSOR_VIEWS_HPP

#include <execution>
#include <future>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include "tensor.hpp"

using namespace std;

namespace microml {

// Adds a constant to every value of a matrix through a view
    class TensorAddScalarView : public BaseTensorUnaryOperatorView {
    public:
        TensorAddScalarView(const shared_ptr<BaseTensor> &tensor, float adjustment)
                : BaseTensorUnaryOperatorView(tensor) {
            this->adjustment = adjustment;
        }

        void printMaterializationPlan() override {
            cout << "TensorAddScalarView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return child->get_val(row, column, channel) + adjustment;
        }

        [[nodiscard]] float get_adjustment() const {
            return adjustment;
        }

    private:
        float adjustment;
    };

// Multiply each element of the tensor by a constant.
    class TensorMultiplyByScalarView : public BaseTensorUnaryOperatorView {
    public:
        TensorMultiplyByScalarView(const shared_ptr <BaseTensor> &tensor, float scale) : BaseTensorUnaryOperatorView(
                tensor) {
            this->scale = scale;
        }

        void printMaterializationPlan() override {
            cout << "TensorMultiplyByScalarView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return scale * child->get_val(row, column, channel);
        }

        [[nodiscard]] float get_scale() const {
            return scale;
        }

    private:
        float scale;
    };

    class TensorValueTransformView : public BaseTensorUnaryOperatorView {
    public:
        TensorValueTransformView(const shared_ptr <BaseTensor> &tensor, function<float(float)> transformFunction)
                : BaseTensorUnaryOperatorView(
                tensor) {
            this->transformFunction = std::move(transformFunction);
        }

        void printMaterializationPlan() override {
            cout << "TensorValueTransformView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return transformFunction(child->get_val(row, column, channel));
        }

    private:
        function<float(float)> transformFunction;
    };

    class TensorValueTransform2View : public BaseTensorUnaryOperatorView {
    public:
        TensorValueTransform2View(const shared_ptr <BaseTensor> &tensor,
                                  function<float(float, vector<double>)> transformFunction,
                                  vector<double> constants) : BaseTensorUnaryOperatorView(
                tensor) {
            this->transformFunction = std::move(transformFunction);
            this->constants = std::move(constants);
        }

        void printMaterializationPlan() override {
            cout << "TensorValueTransform2View{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return transformFunction(child->get_val(row, column, channel), constants);
        }

    private:
        function<float(float, vector<double>)> transformFunction;
        vector<double> constants;
    };

// Change the number of rows and columns, but maintain the same number of elements per channel.
// You cannot change the number of channels in the current implementation.
    class TensorReshapeView : public BaseTensorUnaryOperatorView {
    public:
        TensorReshapeView(const shared_ptr <BaseTensor> &tensor, const size_t rows,
                          const size_t columns) : BaseTensorUnaryOperatorView(tensor) {
            this->rows = rows;
            this->columns = columns;
            this->elements_per_channel = (unsigned long) rows * (unsigned long) columns;
            if (tensor->elements_per_channel() != elements_per_channel) {
                throw exception("A matrix view must be put over a matrix with the same number of elements.");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorReshapeView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return columns;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            const unsigned long position_offset = (row * columns) + column;
            const size_t child_col_count = child->column_count();
            const size_t new_row = position_offset / child_col_count;
            const size_t new_col = position_offset % child_col_count;
            return child->get_val(new_row, new_col, channel);
        }


    private:
        size_t rows;
        size_t columns;
        unsigned long elements_per_channel;
    };

// Converts a 3d tensor into a row vector
    class TensorFlattenToRowView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorFlattenToRowView(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
            this->columns = tensor->size();
        }

        void printMaterializationPlan() override {
            cout << "TensorFlattenToRowView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        size_t row_count() override {
            return 1;
        }

        size_t column_count() override {
            return columns;
        }

        size_t channel_count() override {
            return 1;
        }

        bool readRowsInParallel() override {
            return false;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            if (row != 0 || channel != 0) {
                throw exception("Row Vector has only a single row and channel.");
            }
            return child->get_val(column);
        }


    private:
        size_t columns;
    };

// Converts a 3d tensor into a column vector
    class TensorFlattenToColumnView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorFlattenToColumnView(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(
                tensor) {
            this->rows = tensor->size();
        }

        void printMaterializationPlan() override {
            cout << "TensorFlattenToColumnView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return 1;
        }

        size_t channel_count() override {
            return 1;
        }

        bool readRowsInParallel() override {
            return true;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            if (column != 0 || channel != 0) {
                throw exception("Column Vector has only a single column and channel.");
            }
            return child->get_val(row);
        }


    private:
        size_t rows;
    };

    class TensorTransposeView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorTransposeView(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }

        void printMaterializationPlan() override {
            cout << "TensorTransposeView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        size_t row_count() override {
            return child->column_count();
        }

        size_t column_count() override {
            return child->row_count();
        }

        size_t channel_count() override {
            return child->channel_count();
        }

        bool readRowsInParallel() override {
            return !child->readRowsInParallel();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            // making it obvious that we intend to swap column and row. Compiler will optimize this out.
            const size_t swapped_row = column;
            const size_t swapped_col = row;
            return child->get_val(swapped_row, swapped_col, channel);
        }
    };

// In the current implementation, a tensor is a vector of matrices, and our math is frequently
// interested in each matrix rather than treating the tensor as a whole, so this implementation
// returns the diagonal of each matrix in the tensor.
//
// 0, 1, 2
// 3, 4, 5   becomes  0, 4, 8
// 6, 7, 8
//
// If the tensor has more channels, we do the same thing for each channel.
// If you want to learn more about eiganvalues and diagonalization, and you don't mind
// a lot of math jargon, read here:
// https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
// or here:
// https://mathworld.wolfram.com/MatrixDiagonalization.html
//
// Personally, my linear algebra class was ~25 years ago, and I found this refresher
// useful: https://www.youtube.com/playlist?list=PLybg94GvOJ9En46TNCXL2n6SiqRc_iMB8
// and specifically: https://www.youtube.com/watch?v=WTLl03D4TNA
    class TensorDiagonalView : public BaseTensorUnaryOperatorView {
    public:
        TensorDiagonalView(const shared_ptr <BaseTensor> &tensor, size_t row_offset) : BaseTensorUnaryOperatorView(
                tensor) {
            this->row_offset = row_offset;
            this->is_1d = tensor->row_count() == 1;
            if(!is_1d) {
                // we only have as many columns as there were rows
                this->columns = tensor->row_count() - row_offset;
                // we either have 0 or 1 result row
                this->rows = row_offset < tensor->row_count();
            } else {
                this->columns = tensor->column_count() - row_offset;
                this->rows = this->columns;
            }

        }

        void printMaterializationPlan() override {
            cout << "TensorDiagonalView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        explicit TensorDiagonalView(const shared_ptr <BaseTensor> &tensor)
                : TensorDiagonalView(tensor, 0) {
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return columns;
        }

        bool readRowsInParallel() override {
            return false;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            if( is_1d) {
                if( row + row_offset == column) {
                    child->get_val(0, column, channel);
                }
                return 0.f;
            }
            // we aren't bounds checking, so the caller better make sure that row_count > 0
            return child->get_val(column + row_offset, column, channel);
        }

    private:
        size_t row_offset;
        size_t columns;
        size_t rows;
        bool is_1d;
    };


    class TensorNoOpView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorNoOpView(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {}

        float get_val(size_t row, size_t column, size_t channel) override {
            return child->get_val(row, column, channel);
        }

        void printMaterializationPlan() override {
            cout << "TensorNoOpView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

    private:
    };

    class TensorDotTensorView : public BaseTensorBinaryOperatorView {
    public:
        TensorDotTensorView(const shared_ptr <BaseTensor> &tensor1,
                            const shared_ptr <BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->column_count() != tensor2->row_count()) {
                cout << "[" << tensor1->row_count() << ", " << tensor1->column_count() << ", " << tensor1->channel_count() << "] dot [";
                cout << tensor2->row_count() << ", " << tensor2->column_count() << ", " << tensor2->channel_count() << "]" << endl;
                throw exception("Dot product tensor1.cols must match tensor2.rows in length");
            }
            if (tensor1->channel_count() != tensor2->channel_count()) {
                throw exception("Dot product tensor1.channels must match tensor2.channels in length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorDotTensorView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t row_count() override {
            return child1->row_count();
        }

        size_t column_count() override {
            return child2->column_count();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
//        cout << "getting val: " << row << ", " << column << endl;
            // TODO: I think I should switch to #pragma omp for
            float val = 0;
            for (size_t t1_col = 0; t1_col < child1->column_count(); t1_col++) {
//            cout << "... + "<< child1->get_val(row, t1_col, channel) <<" (" << row << ", " << t1_col << ") * " << child2->get_val(t1_col, column, channel) << "( " << t1_col << ", " << column << ")" << endl;
                val += child1->get_val(row, t1_col, channel) * child2->get_val(t1_col, column, channel);
            }
            return val;
        }
    };

    class TensorMultiplyTensorView : public BaseTensorBinaryOperatorView {
    public:
        TensorMultiplyTensorView(const shared_ptr <BaseTensor> &tensor1,
                                 const shared_ptr <BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1,
                                                                                                        tensor2) {
            if (tensor1->column_count() != tensor2->column_count() || tensor1->row_count() != tensor2->row_count()) {
                cout << "[" << tensor1->row_count() << ", " << tensor1->column_count() << ", " << tensor1->channel_count() << "] * [";
                cout << tensor2->row_count() << ", " << tensor2->column_count() << ", " << tensor2->channel_count() << "]" << endl;
                throw exception("Multiply cols and rows much match in length");
            }
            if (tensor1->channel_count() != tensor2->channel_count()) {
                cout << "[" << tensor1->row_count() << ", " << tensor1->column_count() << ", " << tensor1->channel_count() << "] * [";
                cout << tensor2->row_count() << ", " << tensor2->column_count() << ", " << tensor2->channel_count() << "]" << endl;
                throw exception("Multiply product tensor1.channels must match tensor2.channels in length");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorMultiplyTensorView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t row_count() override {
            return child1->row_count();
        }

        size_t column_count() override {
            return child1->column_count();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
//        cout << "getting val: " << row << ", " << column << endl;
            return child1->get_val(row, column, channel) * child2->get_val(row, column, channel);
        }
    };

    class TensorAddTensorView : public BaseTensorBinaryOperatorView {
    public:
        TensorAddTensorView(const shared_ptr <BaseTensor> &tensor1,
                            const shared_ptr <BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->channel_count() != tensor2->channel_count() || tensor1->row_count() != tensor2->row_count() ||
                tensor1->column_count() != tensor2->column_count()) {
                cout << "[" << tensor1->row_count() << ", " << tensor1->column_count() << ", " << tensor1->channel_count() << "] + [";
                cout << tensor2->row_count() << ", " << tensor2->column_count() << ", " << tensor2->channel_count() << "]" << endl;
                throw exception("You can only add two tensors of the same dimensions together.");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorAddTensorView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t row_count() override {
            return child1->row_count();
        }

        size_t column_count() override {
            return child1->column_count();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return child1->get_val(row, column, channel) + child2->get_val(row, column, channel);
        }
    };

    class TensorMinusTensorView : public BaseTensorBinaryOperatorView {
    public:
        TensorMinusTensorView(const shared_ptr <BaseTensor> &tensor1,
                              const shared_ptr <BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
            if (tensor1->channel_count() != tensor2->channel_count() || tensor1->row_count() != tensor2->row_count() ||
                tensor1->column_count() != tensor2->column_count()) {
                cout << "[" << tensor1->row_count() << ", " << tensor1->column_count() << ", " << tensor1->channel_count() << "] - [";
                cout << tensor2->row_count() << ", " << tensor2->column_count() << ", " << tensor2->channel_count() << "]" << endl;
                throw exception("You can only add two tensors of the same dimensions together.");
            }
        }

        void printMaterializationPlan() override {
            cout << "TensorMinusTensorView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        size_t row_count() override {
            return child1->row_count();
        }

        size_t column_count() override {
            return child1->column_count();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return child1->get_val(row, column, channel) - child2->get_val(row, column, channel);
        }
    };

    class TensorPowerView : public BaseTensorUnaryOperatorView {
    public:
        TensorPowerView(const shared_ptr <BaseTensor> &tensor, const float power) : BaseTensorUnaryOperatorView(
                tensor) {
            this->power = power;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            const float val = child->get_val(row, column, channel);
            return powf(val, power);
        }

        void printMaterializationPlan() override {
            cout << "TensorMinusTensorView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
    private:
        float power;
    };

    class TensorLogView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorLogView(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }
        void printMaterializationPlan() override {
            cout << "TensorLogView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
        float get_val(size_t row, size_t column, size_t channel) override {
            const float val = child->get_val(row, column, channel);
            return log(val);
        }

    private:
    };

    class TensorLog2View : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorLog2View(const shared_ptr <BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }
        void printMaterializationPlan() override {
            cout << "TensorLog2View{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
        float get_val(size_t row, size_t column, size_t channel) override {
            const float val = child->get_val(row, column, channel);
            return log2(val);
        }

    private:
    };

    class TensorRotate180View : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorRotate180View(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
            row_base_value = child->row_count() - 1;
            column_base_value = child->column_count() - 1;
        }
        void printMaterializationPlan() override {
            cout << "TensorRotate180View{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
        float get_val(size_t row, size_t column, size_t channel) override {
            const float val = child->get_val(row_base_value - row, column_base_value - column, channel);
            return val;
        }
    private:
        size_t row_base_value;
        size_t column_base_value;
    };

    class TensorRoundedView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorRoundedView(const shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        }
        void printMaterializationPlan() override {
            cout << "TensorRoundedView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
        float get_val(size_t row, size_t column, size_t channel) override {
            const float val = child->get_val(row, column, channel);
            return round(val);
        }

    private:
    };

    // For a given tensor, sum the all values and place at a specific channel index, while other channels
    // are all zero. This allows us to not only sum the tensors channels into a single channel,
    // but combine the resulting tensor with other tensors.
    class TensorToChannelView : public BaseTensorUnaryOperatorView {
    public:
        TensorToChannelView(const shared_ptr<BaseTensor> &tensor, size_t data_channel_index, size_t number_of_channels) : BaseTensorUnaryOperatorView(tensor) {
            this->data_channel_index = data_channel_index;
            this->number_of_channels = number_of_channels;
        }
        void printMaterializationPlan() override {
            cout << "TensorToChannelView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        size_t channel_count() override {
            return number_of_channels;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            if(channel != data_channel_index) {
                return 0.f;
            }
            float result = 0.f;
            const size_t channels = child->channel_count();
            // TODO: I think I should switch to #pragma omp for
            for( size_t next_channel = 0; next_channel < channels; next_channel++) {
                result += child->get_val(row, column, next_channel);
            }
            return result;
        }
    private:
        size_t data_channel_index;
        size_t number_of_channels;
    };

    class TensorSumChannelsView : public TensorToChannelView {
    public:
        explicit TensorSumChannelsView(const shared_ptr<BaseTensor> &tensor) : TensorToChannelView(tensor, 0, 1) {
        }
        void printMaterializationPlan() override {
            cout << "TensorSumChannelsView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }
    };

    class TensorChannelToTensorView : public BaseTensorUnaryOperatorView {
    public:
        explicit TensorChannelToTensorView(const shared_ptr<BaseTensor> &tensor, size_t channel_offset) : BaseTensorUnaryOperatorView(tensor) {
            this->channel_offset = channel_offset;
        }
        void printMaterializationPlan() override {
            cout << "TensorChannelToChannel{" << row_count() << "," <<column_count()<<",1}->";
            child->printMaterializationPlan();
        }

        size_t channel_count() override {
            return 1;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            if(channel != 0) {
                return 0.f;
            }

            const float val = child->get_val(row, column, channel+channel_offset);
            return val;
        }
    private:
        size_t channel_offset;
    };

    // padding is the amount of extra cells on a given "side" of the matrix
    // so a col_padding of 2 would mean 2 cells to the left that are 0 and 2 cells to the right that are zero.
    // for a total of 4 extra cells in the row.
    class TensorZeroPaddedView : public BaseTensorUnaryOperatorView {
    public:
        TensorZeroPaddedView(const shared_ptr<BaseTensor> &tensor, size_t top_padding, size_t bottom_padding,
                             size_t left_padding, size_t right_padding) : BaseTensorUnaryOperatorView(tensor) {
            this->top_padding = top_padding;
            this->bottom_padding = bottom_padding;
            this->left_padding = left_padding;
            this->right_padding = right_padding;
        }

        void printMaterializationPlan() override {
            cout << "TensorZeroPaddedView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->";
            child->printMaterializationPlan();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            const auto adjusted_row = row - top_padding;
            const auto adjusted_col = column - left_padding;
            // todo: for performance, we can potentially store rowcount as a constant when we make this object.
            if(row < top_padding || adjusted_row >= child->row_count() ||
               column < left_padding || adjusted_col >= child->column_count()) {
                return 0.f;
            }
            const float val = child->get_val(adjusted_row, adjusted_col, channel);
            return val;
        }

        size_t row_count() override {
            const size_t padding = right_padding + left_padding;
            return child->row_count()+padding;
        }

        size_t column_count() override {
            const size_t padding = bottom_padding + top_padding;
            return child->column_count()+padding;
        }
    private:
        size_t top_padding;
        size_t bottom_padding;
        size_t left_padding;
        size_t right_padding;
    };

    class TensorValidCrossCorrelation2dView : public BaseTensorBinaryOperatorView {
    public:
        TensorValidCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : BaseTensorBinaryOperatorView(tensor, kernel) {
            rows = child1->row_count() - child2->row_count() + 1;
            cols = child1->column_count() - child2->column_count() + 1;
        }

        void printMaterializationPlan() override {
            cout << "TensorValidCrossCorrelation2dView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            // iterate over filter
            const auto kernel_rows = child2->row_count();
            const auto kernel_cols = child2->column_count();
            float result = 0.f;
            // TODO: I think I should switch to #pragma omp for collapse(2)
            for(size_t kernel_row = 0; kernel_row < kernel_rows; kernel_row++) {
                for(size_t kernel_col = 0; kernel_col < kernel_cols; kernel_col++) {
                    const auto filter_val = child2->get_val(kernel_row, kernel_col, channel);
                    const auto tensor_val = child1->get_val(row + kernel_row, column + kernel_col, channel);
                    result += filter_val + tensor_val;
                }
            }
            return result;
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return cols;
        }

        size_t channel_count() override {
            return child1->channel_count();
        }
    private:
        size_t rows;
        size_t cols;
    };

    // https://en.wikipedia.org/wiki/Cross-correlation
    // https://en.wikipedia.org/wiki/Two-dimensional_correlation_analysis
    // Having an even number of values in your filter is a little wierd, but I handle it
    // you'll see where I do the rounding, it's so that a 2x2 filter or a 4x4 filter would work.
    // I think most filters are odd numbers because even filters don't make much sense,
    // but I wanted the code to work. The center of your filter is at a halfway point,
    // which is why it's weird.
    class TensorFullCrossCorrelation2dView : public TensorValidCrossCorrelation2dView {
    public:
        TensorFullCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : TensorValidCrossCorrelation2dView(
                make_shared<TensorZeroPaddedView>(tensor,
                                                  (size_t)round(((double)kernel->row_count() - 1) / 2.0),
                                                  (size_t)round(((double)kernel->row_count() - 1) / 2.0),
                                                  (size_t)round(((double)kernel->column_count() - 1) / 2.0),
                                                  (size_t)round(((double)kernel->column_count() - 1) / 2.0)),
                kernel) {
        }

        void printMaterializationPlan() override {
            cout << "TensorFullCrossCorrelation2dView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }
    private:
    };

    // A convolved tensor appears to be equivalent to cross correlation with the filter rotated 180 degrees:
    // https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
    class TensorFullConvolve2dView : public TensorFullCrossCorrelation2dView {
    public:
        TensorFullConvolve2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &kernel)
                : TensorFullCrossCorrelation2dView(tensor, make_shared<TensorRotate180View>(kernel)) {

        }

        void printMaterializationPlan() override {
            cout << "TensorFullConvolve2dView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }
    };
}
#endif //MICROML_TENSOR_VIEWS_HPP
