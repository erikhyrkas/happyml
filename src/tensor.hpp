//
// Created by Erik Hyrkas on 10/25/2022.
//

#ifndef MICROML_TENSOR_HPP
#define MICROML_TENSOR_HPP

#include <execution>
#include <future>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include "quarter_float.hpp"
#include "half_float.hpp"

// TODO:
// * create bit matrix since there are many inputs that are strictly 1s and 0s
//    -- NOTE: the challenge is initializing without a set(). If we allocate with a vector of floats...
//             well, we will end up spending more memory and cpu for a very short lived object.
// * consider improving parallel operations with something like: bool optimized_for_columns_first_iteration() {}
//   true for base matrices because of memory organization and caching, but views may transform the underlying
//   matrix and our iterations would then be slower to iterate by column or row from its perspective. the cpu
//   will automatically pull extra adjacent bytes of the vector into memory, so we want that cached data. if we can.
// * consider if matrices can be made completely immutable.
//   * TL;DR: Semi-immutable matrices, but create an assign(view, matrix_working_memory) function and a is_child(matrix) function
//   * NEW: compromise option -- assign:
//     Make matrices mostly immutable except through an assign function. There can be multiple variations on this function:
//     * view as a parameter with a matrix to hold intermediate results let's us have a place to store temporary results
//       before copying them back into the real matrix. This let's us have at most one copy of working memory for overhead.
//       we don't have to use that working memory if the current matrix isn't a child of the view, since it would be safe
//       to directly update the current matrix.
//     * view as a parameter, no working memory matrix. We need to check if the current matrix is a child of the view.
//       If not, then just copy values from view directly into this matrix. If we are a child. we'd just allocate working
//       memory temporarily and call the first function.
//     * another matrix as a parameter. If it isn't us, then we can start copying. No risk.
//     * we can also have an assign from array, that copies in the values.
//   * argument for immutable:
//     TL;DR: immutable reduces programming complexity and increases safety
//     Views on mutable sources is dangerous if those views are being used to then alter the very underlying source.
//     So, to be safe, current view-based math implementation would have a full copy of the matrix results
//     in memory before it could be assigned the results to a matrix. Well...okay, the results could go to disk instead of to memory.
//     But you can't safely modify a matrix you are simultaneously reading from using a view.
//     You might argue that this need to store a full extra copy of the results in memory rather than modify the original
//     must be wasteful and defeat the purpose of a design that wants to minimize memory footprint. However, you have to
//     remember that in the standard flow of math in some steps, we'd need 2 copies of the matrix in memory
//     if we didn't use views... what's more we lose accuracy every time we do a math step and save to 8-bit. By accumulating
//     all changes so we can do them in 32-bit, and then saving only after all steps are applied, we eliminate a significant
//     amount of rounding. Where exactly would these copies be? 1. Calculating the loss function can result in an extra matrix.
//     2. Not all activation functions can safely write back to the input without an extra intermediate copy or two.
//   * argument against immutable:
//     TL;DR: mutable is faster and less memory
//     We can't reuse a matrix's memory allocation easily and allocations could be in the 10s of gb.
//     Maybe we can safely take in one matrix's allocation into the constructor of another and steal the memory allocation?
//     I think if I did that, I'd use a unique pointer to a raw array and move it. Now we are in dangerous territory that
//     is likely worse than just allowing the matrix to be mutable. A resource pool is complicated and potentially wasteful.
//     Also, not all activation functions require a copy of a matrix... and maybe I'll use more total memory than I saved
//     by being 8-bit.
//     Creating a rule for safety might be wasteful. The big example is adding to the weights. We don't need to directly
//     reference the original weights during calculation of the value we are adding, so what's the harm of reusing the
//     allocation we have and just adding the values we calculated?
//     The other challenge with the matrix being immutable is: how do we populate the initial values? Yes, some matrices
//     start as random or all zeros and math happens nad you can initialize future matrices from the outputs of views.
//     What about the matrices that represent data sets? Do you have to allocate an array that then you copy values
//     from and into the matrix? I suppose the array buffer could be reused if it is fixed size.
// * if immutable:
//   * comment out non-view-based matrix math functions external to class. this should be done through a view.
//   * comment out matrix class methods that adjust values as they should be done through a view
//   * there could be a view that gives us a similar pseudo random value for a matrix coordinate that doesn't require
//     allocating a full matrix. This is useful if we aren't reusing the original weight matrix and let's us skip
//     an allocation on the first pass. This pseudo random number would have more granularity than the 8-bit float
//     we would otherwise start with. At the same time, I'm not sure how much the starting values matter.
//

/*
 * A few basic design decisions here:
 *
 * Originally, I let the Matrix be resized and reshaped, but I wanted the shape to be immutable to support a
 * view over the original object without that view breaking later if somebody changed the shape. The view is
 * an important performance improvement to allow us to change the arrangement of columns and rows without copying
 * the full object and using a lot of memory.
 *
 * I didn't want any methods creating objects that needed to be managed beyond the constructor. I briefly had
 * a reshape method and then decided to turn it into a constructor. This makes memory ownership more apparent
 * and allows the caller to decide how to do that memory management.
 *
 * I decided to use vectors rather than an array. By using a vector holding other vectors, the code would never make
 * a single huge allocation. This may turn out to be a good or bad decision. However, I think that the pros of being
 * able to have a considerably larger total matrix will outweigh any challenges around memory fragmentation or possible
 * performance issues around many allocations. (I'm not even sure this will be slower in the case of large matrices
 * since making monolithic allocations would be slow. There's going to be a break-even point where this solution is
 * better even though for small allocations an array would definitely be better.)
 *
 * Since I'm using vectors of vectors, I do wonder whether there will be performance implications with huge number of
 * operations accessing individual elements.
 *
 * Since I don't allow resizing or reshaping, I don't provide an empty constructor.
 *
 * I do provide a number of handy methods to change the values of the matrix. One of which is random(). I'm not
 * certain that this implementation will perform well at scale. We'll see.
 */
using namespace std;

namespace microml {

    void wait_for_futures(queue <future<void>> &futures) {
        while (!futures.empty()) {
            futures.front().wait();
            futures.pop();
        }
    }

    class BaseTensor : public enable_shared_from_this<BaseTensor> {
    public:
        virtual size_t row_count() = 0;

        virtual size_t column_count() = 0;

        virtual size_t channel_count() = 0;

        virtual bool isMaterialized() {
            return false;
        }

        virtual void printMaterializationPlanLine() {
            printMaterializationPlan();
            cout << endl;
        }

        virtual void printMaterializationPlan() = 0;
        // fastest read is generally along columns because of how memory is organized,
        // but we can't do a parallel read if there's only one row.
        virtual bool readRowsInParallel() {
            return (row_count() > 1);
        }

        virtual bool contains(const shared_ptr<BaseTensor> &other) {
            return other == shared_from_this();
        }

        unsigned long size() {
            return row_count() * column_count() * channel_count();
        }

        unsigned long elements_per_channel() {
            return row_count() * column_count();
        }

        virtual float get_val(size_t row, size_t column, size_t channel) = 0;

        virtual vector <size_t> getShape() {
            return {row_count(), column_count(), channel_count()};
        }

        float get_val(const unsigned long position_offset) {
            const size_t cols = column_count();
            const unsigned long matrix_size = cols * row_count();
            const size_t new_channel = position_offset / matrix_size;
            const unsigned long matrix_elements = position_offset % matrix_size;
            const size_t new_row = matrix_elements / cols;
            const size_t new_col = matrix_elements % cols;
            return get_val(new_row, new_col, new_channel);
        }

        double product() {
            double result = 1.0;
            const size_t max_rows = row_count();
            const size_t max_cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < max_rows; row++) {
                    for (size_t col = 0; col < max_cols; col++) {
                        result *= get_val(row, col, channel);
                    }
                }
            }
            return result;
        }

        double sum() {
            double result = 0.0;
            const size_t max_rows = row_count();
            const size_t max_cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < max_rows; row++) {
                    for (size_t col = 0; col < max_cols; col++) {
                        result += get_val(row, col, channel);
                    }
                }
            }
            return result;
        }

        float max() {
            float result = -INFINITY;
            const size_t max_rows = row_count();
            const size_t max_cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < max_rows; row++) {
                    for (size_t col = 0; col < max_cols; col++) {
                        result = std::max(result, get_val(row, col, channel));
                    }
                }
            }
            return result;
        }

        float min() {
            float result = INFINITY;
            const size_t max_rows = row_count();
            const size_t max_cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < max_rows; row++) {
                    for (size_t col = 0; col < max_cols; col++) {
                        result = std::min(result, get_val(row, col, channel));
                    }
                }
            }
            return result;
        }

        pair<float, float> range() {
            float min_result = INFINITY;
            float max_result = -INFINITY;
            const size_t max_rows = row_count();
            const size_t max_cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < max_rows; row++) {
                    for (size_t col = 0; col < max_cols; col++) {
                        const auto val = get_val(row, col, channel);
                        min_result = std::min(min_result, val);
                        max_result = std::max(max_result, val);
                    }
                }
            }
            return {min_result, max_result};
        }

        size_t max_index(size_t channel, size_t row) {
            size_t result = 0;
            float current_max = -INFINITY;
            const size_t max_cols = column_count();
            for (size_t col = 0; col < max_cols; col++) {
                float next_val = get_val(row, col, channel);
                if (next_val > current_max) {
                    current_max = next_val;
                    result = col;
                }
            }
            return result;
        }

        size_t min_index(size_t channel, size_t row) {
            size_t result = 0;
            float current_min = INFINITY;
            const size_t max_cols = column_count();
            for (size_t col = 0; col < max_cols; col++) {
                float next_val = get_val(row, col, channel);
                if (next_val < current_min) {
                    current_min = next_val;
                    result = col;
                }
            }
            return result;
        }

        vector <size_t> max_indices(size_t channel, size_t row) {
            vector<size_t> result;
            float current_max = -INFINITY;
            const size_t max_cols = column_count();
            for (size_t col = 0; col < max_cols; col++) {
                float next_val = get_val(row, col, channel);
                if (next_val > current_max) {
                    current_max = next_val;
                    result.clear();
                    result.push_back(col);
                } else if (next_val == current_max) {
                    result.push_back(col);
                }
            }
            return result;
        }

        vector <size_t> min_indices(size_t channel, size_t row) {
            vector<size_t> result;
            float current_min = INFINITY;
            const size_t max_cols = column_count();
            for (size_t col = 0; col < max_cols; col++) {
                float next_val = get_val(row, col, channel);
                if (next_val < current_min) {
                    current_min = next_val;
                    result.clear();
                    result.push_back(col);
                } else if (next_val == current_min) {
                    result.push_back(col);
                }
            }
            return result;
        }

        float arithmetic_mean() {
            // This is your basic average.
            //
            // Calculating a mean is something every teenager does. You sum up the numbers and divide by
            // the number of elements. For example: (1 + 2 + 3) / 3  =  6 / 3  =  2
            // We have a large number of elements in a matrix where we would overflow if we summed them up,
            // so we have to use a different formula.
            //
            // (Overflow is where you try to put a bigger number than the data type can handle and get wrong results.)
            //
            // for each offset:
            //   average = average + (val[offset] - average)/(offset+1)
            double average = 0;
            double index = 0;
            const size_t rows = row_count();
            const size_t cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < cols; col++) {
                        index++;
                        const double val = get_val(row, col, channel);
                        average += (val - average) / index;
                    }
                }
            }

            return (float) average;
        }

        float geometric_mean() {
            // Geometric mean is found by multiplying your list of numbers and then taking the n-th root of it, where
            // n is how many numbers we have.
            //
            // Much like calculating the mean, it is easy to calculate on paper, but the trick is doing
            // it for lots of numbers without overflow. I'm not sure if this solution is going to work
            // in all necessary situations. I may have to revisit it later.
            //
            // It's important to remember that you cannot use zero or negative numbers in a geometric mean.
            //
            // You could work around the zero limitation. For example, you could convert zeros to ones.
            // However, the results won't be a formal geometric mean, and the caller could do that swap themselves
            // to get their own version of geometric mean -- since nothing I did here would be formally correct.
            //
            // We could possibly handle zeros and negative numbers by converting the input into a percentages of (something),
            // then multiply the result (which would now be a percentage) by (something). I experimented
            // some and the results were almost right but since they weren't truly right, I left it out for now.
            // And what's more, this is something that the caller could do before calling this method if that approximation
            // was good enough.
            //
            // What I understand of negative number handling:
            // Positive numbers should be a percentage above 1, where negative numbers a percentage below 1. Then
            // after you do the standard calculations subtract 1 and expand that percentage back into the original scale.
            //
            // I think where I was struggling with was "percentage of what?" Percentage of the biggest number?
            // Percentage of only the positive numbers (seems problematic)? The example I saw used the percentage of the
            // previous number in the series, which seems problematic for a matrix...what is "previous"? And
            // how could that be the right way to do it? Heck, what is previous of the first number?
            //
            // To me, it seemed like it should be arbitrary what I did a percentage of, right? If I had the numbers
            // [10, -1000] and I decided to divide by 10,000, I'd get [0.001, -0.1], which would then become [1.001, 0.9].
            // I felt like the results should be the same if my scale was anything. This did not prove to be true.
            //
            // How did I know the negative number solution didn't work correctly? Because when that solution was applied to
            // only positive numbers, the results didn't match the simple formula that I know to be correct. The results
            // were close, but off by amounts that were too much to possibly be right.
            //
            // This is what I tried last:
            //        double scale_val = max(abs(max()), abs(min()))*10.0;
            //        double sum = 0;
            //        double index = 0;
            //        for (size_t row = 0; row < row_count(); row++) {
            //            for (size_t col = 0; col < col_count(); col++) {
            //                const double val = 1.0+(((double)get_val(row, col))/scale_val);
            //                index++;
            //                sum += log(val);
            //            }
            //        }
            //        sum /= index;
            //        return (float)((exp(sum) - 1.0)*scale_val);
            //
            // This wasn't my only attempt. I tried shifting all numbers to be one or above, then scaling by the range.
            // This also gave nearly correct, but still incorrect answers. I've already forgotten more about what other
            // attempts I made, but I spent too much time on it for something that I don't have an immediate need for
            // and clearly don't fully understand.
            double sum = 0;
            double index = 0;
            const size_t rows = row_count();
            const size_t cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < cols; col++) {
                        const double val = ((double) get_val(row, col, channel));
                        index++;
                        if (val <= 0) {
                            return NAN;
                        }
                        sum += log(val);
                    }
                }
            }
            sum /= index;
            return (float) exp(sum);
        }

        void print() {
            print(cout);
        }

        void print(ostream &out) {
            if(channel_count() > 1) {
                out << "[";
            }
            out << setprecision(3) << fixed << endl;
            const size_t rows = row_count();
            const size_t cols = column_count();
            const size_t max_channels = channel_count();
            for (size_t channel = 0; channel < max_channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    if( rows > 1) {
                        cout << "|";
                    } else {
                        cout << "[";
                    }
                    string delim;
                    for (size_t col = 0; col < cols; col++) {
                        out << delim << get_val(row, col, channel);
                        delim = ", ";
                    }
                    if( rows > 1) {
                        out << "|" << endl;
                    } else {
                        out << "]" << endl;
                    }
                }
            }
            if(channel_count() > 1) {
                out << "]";
            }
        }
    };

// This abstract class lets us build float tensors and bit tensors as well and use them interchangeably.
    class BaseAssignableTensor : public BaseTensor {
    public:
        virtual void assign(const shared_ptr<BaseTensor> &other) = 0;

        virtual void assign(const shared_ptr<BaseTensor> &other, const shared_ptr<BaseAssignableTensor> &working_memory) = 0;

        virtual bool isMaterialized() {
            return true;
        }
    };

// The full tensor is backed by a 32-bit float. This exists because our input into our models may
// require accurate representations, and I don't think they'll ever be too big to fit in memory.
// There may also be final dense layers that have few enough neurons feeding it that a full tensor
// may work.
    class FullTensor : public BaseAssignableTensor {
    public:
        FullTensor(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        explicit FullTensor(const shared_ptr<BaseTensor> &original)
                : FullTensor(original->row_count(),
                             original->column_count(),
                             original->channel_count()) {
            do_assign(original);
        }

        explicit FullTensor(const vector<float> &values) : FullTensor(1, values.size(), 1) {
            size_t col = 0;
            for (float const &val: values) {
                set_val(0, col, 0, val);
                col++;
            }
        }


        // get a weird warning here that CLion can't resolve constructor. I believe this is a bug with CLion itself:
        // https://youtrack.jetbrains.com/issue/CPP-24510/Bad-detection-of-Constructor-is-not-implemented
        explicit FullTensor(const vector<vector<vector<float>>> &values) :
                FullTensor(values[0].size(), values[0][0].size(), values.size()) {
            size_t channel_index = 0;
            for (const vector <vector<float>> &next_channel: values) {
                size_t row_index = 0;
                for (const vector<float> &next_row: next_channel) {
                    size_t col_index = 0;
                    for (float val: next_row) {
                        set_val(row_index, col_index, channel_index, val);
                        col_index++;
                    }
                    row_index++;
                }
                channel_index++;
            }
        }

        void assign(const shared_ptr<BaseTensor> &other, const shared_ptr<BaseAssignableTensor> &working_memory) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
//                cout << "ASSIGN: making copy to working memory -- needs tests" << endl;
                working_memory->assign(other);
                do_assign(working_memory);
            } else {
//                cout << "ASSIGN: no copy to working memory -- needs tests" << endl;
                do_assign(other);
            }
        }

        // will assign the values from other to this tensor, but if the other tensor is
        // a view that contains us, then we avoid data corruption by copying to a temporary
        // tensor first.
        // this could possibly be optimized by leveraging the newly allocated tensor's
        // internal values directly, but our current data elements are vectors, not a pointer
        // to vectors.
        void assign(const shared_ptr<BaseTensor> &other) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
//                cout << "ASSIGN: making copy -- needs tests" << endl;
                auto temp = make_shared<FullTensor>(other);
//            do_assign(*temp); -- let's just steal the memory:
                data = std::move(temp->data);
            } else {
//                cout << "ASSIGN: no copy -- needs tests" << endl;
                do_assign(other);
            }
        }

        size_t channel_count() override {
            return data.size();
        }

        size_t row_count() override {
            if (data.empty()) {
                return 0;
            }
            return data[0].size();
        }

        size_t column_count() override {
            if (data.empty() || data[0].empty()) {
                return 0;
            }
            return data[0][0].size();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return data.at(channel).at(row).at(column);
        }

        void printMaterializationPlan() override {
            cout << "FullTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }
    private:
        vector <vector<vector < float>>> data;

        void do_assign(const shared_ptr<BaseTensor> &other) {
            if (other->row_count() != row_count() && other->channel_count() != channel_count() &&
                other->column_count() != column_count()) {
                throw exception("A tensor cannot be assigned from another tensor with a different shape");
            }

            const size_t columns = column_count();
            const size_t rows = row_count();
            const size_t channels = channel_count();
            if (rows <= 1 && columns < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        populate_by_col(other, this, row, columns, channel);
                    }
                }
            } else if(columns <= 1 && rows < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t col = 0; col < columns; col++) {
                        populate_by_row(other, this, rows, col, channel);
                    }
                }
            } else {
                queue<future<void>> futures;
                const size_t wait_amount = 8096;
                if( readRowsInParallel() ) {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t row = 0; row < rows; row++) {
                            auto next_async = async(launch::async, populate_by_col, other, this, row, columns,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                } else {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t col = 0; col < columns; col++) {
                            auto next_async = async(launch::async, populate_by_row, other, this, rows, col,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                }
                wait_for_futures(futures);
            }
        }


        static void populate_by_col(const shared_ptr<BaseTensor> &source,
                                    FullTensor *dest,
                                    size_t row,
                                    size_t max_cols,
                                    size_t channel) {
            for (size_t col = 0; col < max_cols; col++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        static void populate_by_row(const shared_ptr<BaseTensor> &source,
                                    FullTensor *dest,
                                    size_t max_rows,
                                    size_t col,
                                    size_t channel) {
            for (size_t row = 0; row < max_rows; row++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        inline void set_val(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = val;
        }
    };


// TODO: Okay, so I clearly need another layer of abstraction or to template the BaseAssignableTensor, but
// that'll be another day.
// Pixel Tensor holds a value between 0.0f and 1.0f with an even distribution in 256 increments (8-bits.)
// This is a compact representation useful for images, but also for other data that has an evenly distributed
// range of values between 0 and 1 with a similar granularity.
// The quarter tensor with a bias of 14 is capable of a similar representation, but the distribution of values isn't
// even. This Tensor is also faster than the quarter tensor because far less math needs to happen to map between
// float and 8-bits.
    class PixelTensor : public BaseAssignableTensor {
    public:
        PixelTensor(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        explicit PixelTensor(const shared_ptr<BaseTensor> &original)
                : PixelTensor(original->row_count(),
                              original->column_count(),
                              original->channel_count()) {
            do_assign(original);
        }

        // If you use this constructor, you've already wasted a lot of memory.
        // Maybe you can just use a full tensor?
        explicit PixelTensor(const vector<float> &values) : PixelTensor(1, values.size(), 1) {
            size_t col = 0;
            for (float const &val: values) {
                set_val(0, col, 0, val);
                col++;
            }
        }

        // see the note by FullTensor about the CLion warning bug.
        // If you use this constructor, you've already wasted a lot of memory.
        // Maybe you can just use a full tensor?
        explicit PixelTensor(const vector<vector<vector<float>>> &values) :
                PixelTensor(values[0].size(), values[0][0].size(), values.size()) {
            size_t channel_index = 0;
            for (const auto &next_channel: values) {
                size_t row_index = 0;
                for (const auto &next_row: next_channel) {
                    size_t col_index = 0;
                    for (float val: next_row) {
                        set_val(row_index, col_index, channel_index, val);
                        col_index++;
                    }
                    row_index++;
                }
                channel_index++;
            }
        }

        void assign(const shared_ptr<BaseTensor> &other, const shared_ptr<BaseAssignableTensor> &working_memory) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                cout << "ASSIGN: making copy to working memory -- needs tests" << endl;
                working_memory->assign(other);
                do_assign(working_memory);
            } else {
                cout << "ASSIGN: no copy to working memory -- needs tests" << endl;
                do_assign(other);
            }
        }

        // will assign the values from other to this tensor, but if the other tensor is
        // a view that contains us, then we avoid data corruption by copying to a temporary
        // tensor first.
        // this could possibly be optimized by leveraging the newly allocated tensor's
        // internal values directly, but our current data elements are vectors, not a pointer
        // to vectors.
        void assign(const shared_ptr<BaseTensor> &other) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                cout << "ASSIGN: making copy -- needs tests" << endl;
                auto temp = make_shared<PixelTensor>(other);
//            do_assign(*temp); -- let's just steal the memory:
                data = std::move(temp->data);
            } else {
                cout << "ASSIGN: no copy -- needs tests" << endl;
                do_assign(other);
            }
        }

        size_t channel_count() override {
            return data.size();
        }

        size_t row_count() override {
            if (data.empty()) {
                return 0;
            }
            return data[0].size();
        }

        size_t column_count() override {
            if (data.empty() || data[0].empty()) {
                return 0;
            }
            return data[0][0].size();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return ((float) data.at(channel).at(row).at(column)) / 255.f;
        }
        void printMaterializationPlan() override {
            cout << "PixelTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }
    private:
        vector <vector<vector < uint8_t>>>
        data;

        void do_assign(const shared_ptr<BaseTensor> &other) {
            if (other->row_count() != row_count() && other->channel_count() != channel_count() &&
                other->column_count() != column_count()) {
                throw exception(
                        "A tensor cannot be assigned from another tensor with a different shape");
            }

            const size_t columns = column_count();
            const size_t rows = row_count();
            const size_t channels = channel_count();
            if (rows <= 1 && columns < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        populate_by_col(other, this, row, columns, channel);
                    }
                }
            } else if(columns <= 1 && rows < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t col = 0; col < columns; col++) {
                        populate_by_row(other, this, rows, col, channel);
                    }
                }
            } else {
                queue<future<void>> futures;
                const size_t wait_amount = 8096;
                if( readRowsInParallel()) {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t row = 0; row < rows; row++) {
                            auto next_async = async(launch::async, populate_by_col, other, this, row, columns,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                } else {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t col = 0; col < columns; col++) {
                            auto next_async = async(launch::async, populate_by_row, other, this, rows, col,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                }
                wait_for_futures(futures);
            }
        }

        static void populate_by_col(const shared_ptr<BaseTensor> &source,
                                    PixelTensor *dest,
                                    size_t row,
                                    size_t max_cols,
                                    size_t channel) {
            for (size_t col = 0; col < max_cols; col++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        static void populate_by_row(const shared_ptr<BaseTensor> &source,
                                    PixelTensor *dest,
                                    size_t max_rows,
                                    size_t col,
                                    size_t channel) {
            for (size_t row = 0; row < max_rows; row++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        inline void set_val(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = (uint8_t) (std::max(0.0f, std::min(val, 1.0f)) * 255);
        }
    };

    class QuarterTensor : public BaseAssignableTensor {
    public:
        QuarterTensor(const size_t rows, const size_t columns, const size_t channels, const int bias) {
            this->bias = bias;
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        explicit QuarterTensor(const shared_ptr<BaseTensor> &original, const int bias)
                : QuarterTensor(original->row_count(),
                                original->column_count(),
                                original->channel_count(),
                                bias) {
            do_assign(original);
        }

        QuarterTensor(const vector<float> &values, const int bias)
                : QuarterTensor(1, values.size(), 1, bias) {
            size_t col = 0;
            for (float const &val: values) {
                set_val(0, col, 0, val);
                col++;
            }
        }

        QuarterTensor(const vector <vector<float>> &values, const int bias)
                : QuarterTensor( values.size(), values.at(0).size(), 1, bias) {
            for (size_t row = 0; row < values.size(); row++) {
                for (size_t col = 0; col < values[row].size(); col++) {
                    const float val = values.at(row).at(col);
                    set_val(row, col, 0, val);
                }
            }
        }

        size_t channel_count() override {
            return data.size();
        }

        size_t row_count() override {
            if (data.empty()) {
                return 0;
            }
            return data[0].size();
        }

        size_t column_count() override {
            if (data.empty() || data[0].empty()) {
                return 0;
            }
            return data[0][0].size();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return quarter_to_float(data.at(channel).at(row).at(column), bias);
        }


        [[nodiscard]] int get_bias() const {
            return bias;
        }

        // We're potentially dealing with huge amounts of memory, and we don't want to
        // allocate and reallocate. If the tensor we are assigning from contains ourselves
        // we'd corrupt the data if we wrote to the tensor while reading from it, so we
        // need working memory to hold the temporary results.
        void assign(const shared_ptr<BaseTensor> &other, const shared_ptr<BaseAssignableTensor> &working_memory) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                working_memory->assign(other);
                do_assign(working_memory);
            } else {
                do_assign(other);
            }
        }

        // will assign the values from other to this tensor, but if the other tensor is
        // a view that contains us, then we avoid data corruption by copying to a temporary
        // tensor first.
        // this could possibly be optimized by leveraging the newly allocated tensor's
        // internal values directly, but our current data elements are vectors, not a pointer
        // to vectors.
        void assign(const shared_ptr<BaseTensor> &other) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                cout << "ASSIGN: making copy -- needs tests" << endl;
                auto temp = make_shared<QuarterTensor>(other, bias);
//            do_assign(*temp); -- let's just steal the memory:
                data = std::move(temp->data);
            } else {
                cout << "ASSIGN: no copy -- needs tests" << endl;
                do_assign(other);
            }
        }

        void printMaterializationPlan() override {
            cout << "QuarterTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }
    private:
        vector<vector<vector<quarter>>> data;
        int bias;

        void do_assign(const shared_ptr<BaseTensor> &other) {
            if (other->row_count() != row_count() && other->channel_count() != channel_count() &&
                other->column_count() != column_count()) {
                throw exception(
                        "A tensor cannot be assigned from another tensor with a different shape");
            }

            const size_t columns = column_count();
            const size_t rows = row_count();
            const size_t channels = channel_count();
            if (rows <= 1 && columns < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        populate_by_col(other, this, row, columns, channel);
                    }
                }
            } else if(columns <= 1 && rows < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t col = 0; col < columns; col++) {
                        populate_by_row(other, this, rows, col, channel);
                    }
                }
            } else {
                queue<future<void>> futures;
                const size_t wait_amount = 8096;
                if( readRowsInParallel()) {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t row = 0; row < rows; row++) {
                            auto next_async = async(launch::async, populate_by_col, other, this, row, columns,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                } else {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t col = 0; col < columns; col++) {
                            auto next_async = async(launch::async, populate_by_row, other, this, rows, col,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                }
                wait_for_futures(futures);
            }
        }


        static void populate_by_col(const shared_ptr<BaseTensor> &source,
                                    QuarterTensor *dest,
                                    size_t row,
                                    size_t max_cols,
                                    size_t channel) {
//            cout << "pop by col" <<endl;
            for (size_t col = 0; col < max_cols; col++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        static void populate_by_row(const shared_ptr<BaseTensor> &source,
                                    QuarterTensor *dest,
                                    size_t max_rows,
                                    size_t col,
                                    size_t channel) {
//            cout << "pop by row" <<endl;
            for (size_t row = 0; row < max_rows; row++) {
                const float source_val = source->get_val(row, col, channel);
                dest->set_val(row, col, channel,source_val);
//                const float dest_val = dest->get_val(row, col, channel);
//                cout << "conversion: "<< source_val << " -> " << dest_val << endl;
            }
        }

        // Don't assign values directly to a tensor. If you have specific values for specific entries,
        // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
        // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
        // for the final result.
        inline void set_val(size_t row, size_t column, size_t channel, float val) {
            // used only during a test for a breakpoint
//            const float qv = quarter_to_float(float_to_quarter(val, bias, offset), bias, offset);
//            if(std::abs(val-qv) > 0.014 ){
//                cout  << endl << "major precision loss: " << val << " -> " << qv << endl;
//                // in 8000 epochs for the xor test, this didn't happen. most error is below 0.01 with the
//                // rare error between 0.01 and 0.014
//                // where I think this is problematic is over 96000 individual numbers in that original
//                // 8000 epochs each being 0.01 off adds up to significant error.
//            }
            data.at(channel).at(row).at(column) = float_to_quarter(val, bias);
        }
    };


    class HalfTensor : public BaseAssignableTensor {
    public:
        HalfTensor(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        explicit HalfTensor(const shared_ptr<BaseTensor> &original)
                : HalfTensor(original->row_count(),
                                original->column_count(),
                                original->channel_count()) {
            do_assign(original);
        }

        explicit HalfTensor(const vector<float> &values)
                : HalfTensor(1, values.size(), 1) {
            size_t col = 0;
            for (float const &val: values) {
                set_val(0, col, 0, val);
                col++;
            }
        }

        explicit HalfTensor(const vector <vector<float>> &values)
                : HalfTensor( values.size(), values.at(0).size(), 1) {
            for (size_t row = 0; row < values.size(); row++) {
                for (size_t col = 0; col < values[row].size(); col++) {
                    const float val = values.at(row).at(col);
                    set_val(row, col, 0, val);
                }
            }
        }

        size_t channel_count() override {
            return data.size();
        }

        size_t row_count() override {
            if (data.empty()) {
                return 0;
            }
            return data[0].size();
        }

        size_t column_count() override {
            if (data.empty() || data[0].empty()) {
                return 0;
            }
            return data[0][0].size();
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return half_to_float(data.at(channel).at(row).at(column));
        }

        // We're potentially dealing with huge amounts of memory, and we don't want to
        // allocate and reallocate. If the tensor we are assigning from contains ourselves
        // we'd corrupt the data if we wrote to the tensor while reading from it, so we
        // need working memory to hold the temporary results.
        void assign(const shared_ptr<BaseTensor> &other, const shared_ptr<BaseAssignableTensor> &working_memory) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                working_memory->assign(other);
                do_assign(working_memory);
            } else {
                do_assign(other);
            }
        }

        // will assign the values from other to this tensor, but if the other tensor is
        // a view that contains us, then we avoid data corruption by copying to a temporary
        // tensor first.
        // this could possibly be optimized by leveraging the newly allocated tensor's
        // internal values directly, but our current data elements are vectors, not a pointer
        // to vectors.
        void assign(const shared_ptr<BaseTensor> &other) override {
            if (other == shared_from_this()) {
                return; //assignment to self is pointless and expensive
            }
            if (contains(other)) {
                auto temp = make_shared<HalfTensor>(other);
//            do_assign(*temp); -- let's just steal the memory:
                data = std::move(temp->data);
            } else {
                do_assign(other);
            }
        }

        void printMaterializationPlan() override {
            cout << "HalfTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }

    private:
        vector<vector<vector<half>>> data;

        void do_assign(const shared_ptr<BaseTensor> &other) {
            if (other->row_count() != row_count() && other->channel_count() != channel_count() &&
                other->column_count() != column_count()) {
                throw exception(
                        "A tensor cannot be assigned from another tensor with a different shape");
            }

            const size_t columns = column_count();
            const size_t rows = row_count();
            const size_t channels = channel_count();
            if (rows <= 1 && columns < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        populate_by_col(other, this, row, columns, channel);
                    }
                }
            } else if(columns <= 1 && rows < 10000) {
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t col = 0; col < columns; col++) {
                        populate_by_row(other, this, rows, col, channel);
                    }
                }
            } else {
                queue<future<void>> futures;
                const size_t wait_amount = 8096;
                if( readRowsInParallel()) {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t row = 0; row < rows; row++) {
                            auto next_async = async(launch::async, populate_by_col, other, this, row, columns,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                } else {
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t col = 0; col < columns; col++) {
                            auto next_async = async(launch::async, populate_by_row, other, this, rows, col,
                                                    channel);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait_for_futures(futures);
                            }
                        }
                    }
                }
                wait_for_futures(futures);
            }
        }


        static void populate_by_col(const shared_ptr<BaseTensor> &source,
                                    HalfTensor *dest,
                                    size_t row,
                                    size_t max_cols,
                                    size_t channel) {
//            cout << "pop by col" <<endl;
            for (size_t col = 0; col < max_cols; col++) {
                dest->set_val(row, col, channel, source->get_val(row, col, channel));
            }
        }

        static void populate_by_row(const shared_ptr<BaseTensor> &source,
                                    HalfTensor *dest,
                                    size_t max_rows,
                                    size_t col,
                                    size_t channel) {
//            cout << "pop by row" <<endl;
            for (size_t row = 0; row < max_rows; row++) {
                const float source_val = source->get_val(row, col, channel);
                dest->set_val(row, col, channel,source_val);
//                const float dest_val = dest->get_val(row, col, channel);
//                cout << "conversion: "<< source_val << " -> " << dest_val << endl;
            }
        }

        // Don't assign values directly to a tensor. If you have specific values for specific entries,
        // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
        // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
        // for the final result.
        inline void set_val(size_t row, size_t column, size_t channel, float val) {
            // used only during a test for a breakpoint
//            const float qv = quarter_to_float(float_to_quarter(val, bias, offset), bias, offset);
//            if(std::abs(val-qv) > 0.014 ){
//                cout  << endl << "major precision loss: " << val << " -> " << qv << endl;
//                // in 8000 epochs for the xor test, this didn't happen. most error is below 0.01 with the
//                // rare error between 0.01 and 0.014
//                // where I think this is problematic is over 96000 individual numbers in that original
//                // 8000 epochs each being 0.01 off adds up to significant error.
//            }
            data.at(channel).at(row).at(column) = float_to_half(val);
        }
    };


// If you can represent a tensor as a function, we don't have to allocate gigabytes of memory
// to hold it. You already have a compact representation of it.
    class TensorFromFunction : public BaseTensor {
    public:
        TensorFromFunction(function<float(size_t, size_t, size_t)> tensorFunction, size_t rows, size_t cols,
                           size_t channels) {
            this->tensorFunction = std::move(tensorFunction);
            this->rows = rows;
            this->cols = cols;
            this->channels = channels;
        }

        void printMaterializationPlan() override {
            cout << "TensorFromFunction{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return cols;
        }

        size_t channel_count() override {
            return channels;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return tensorFunction(row, column, channel);
        }

    private:
        function<float(size_t, size_t, size_t)> tensorFunction;
        size_t rows;
        size_t cols;
        size_t channels;
    };

// Our requirements are:
// We need a reasonably well distributed set of numbers across a range that can be accessed in
// a thread safe way and produce the same value each time for a given row/column/channel and seed regardless
// of the number of times we access them.
//
// The primary use case for this tensor is a means of initializing weights. I think the math works out no matter
// what weights we start with (my intuition is that it would be inefficient to start with the same value for every
// weight -- but I can't recall if this is a strict requirement.)
//
// A standard random function might give more random results, except that it would require we save the values in memory
// in order to preserve it across multiple requests, and we'd have to generate those random numbers in a single
// threaded (or other special means) to have repeatable. This is very hard to do without putting special restrictions
// on how the code is used or how it is run, because we'd need to reset the state at precise times and that makes
// this tensor different from other tensor classes who don't have such a requirement, and it would still be tricky
// to manage in a multithreaded way. I spent way too much time trying to solve this elegantly with xorshift, and
// I am moving on.
//
// So, the solution below is a very rough pseudo random generator. It's not really random, in the sense that
// the numbers aren't easily predictable. It's mostly simple (and fast) math that stays within the boundaries given.
    class TensorFromRandom : public BaseTensor {
    public:
        TensorFromRandom(size_t rows, size_t cols, size_t channels, float min_value, float max_value, uint32_t seed) {
            this->rows = rows;
            this->cols = cols;
            this->channel_size = (double) rows * (double) cols;
            this->channels = channels;
            this->seed = seed;
            this->min_value = std::min(min_value, max_value);
            this->max_value = std::max(max_value, min_value);
            this->range = fabs(max_value - min_value);
            this->range_const = range / 2.71828;
            this->seed_const = (std::min(seed, (uint32_t) 1) * range_const) / 3.14159265358979323846;
        }

        TensorFromRandom(size_t rows, size_t cols, size_t channels, int bias) :
                TensorFromRandom(rows, cols, channels, quarter_to_float(QUARTER_MIN, bias),
                                 quarter_to_float(QUARTER_MAX, bias), 42) {
        }

        TensorFromRandom(size_t rows, size_t cols, size_t channels, int bias, uint32_t seed) :
                TensorFromRandom(rows, cols, channels, quarter_to_float(QUARTER_MIN, bias),
                                 quarter_to_float(QUARTER_MAX, bias), seed) {
        }

        void printMaterializationPlan() override {
            cout << "TensorFromRandom{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return cols;
        }

        size_t channel_count() override {
            return channels;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            // nothing magical here... I'm finding an offset, expanding it by a massive amount relative to the range,
            // then forcing it into a range. I picked a few constants that I felt gave a reasonable looking distribution.
            const double offset = (((double) channel * channel_size) + ((double) row * (double) cols) +
                                   (((double) column + 1.0) * range_const) + seed_const) * 3.14159265358979323846;
            return (float) (max_value - fmod(offset, range));
        }

        [[nodiscard]] float get_min_value() const {
            return min_value;
        }

        [[nodiscard]] float get_max_value() const {
            return max_value;
        }

        [[nodiscard]] uint32_t get_seed() const {
            return seed;
        }

    private:
        size_t rows;
        size_t cols;
        size_t channels;
        double channel_size;
        float min_value;
        float max_value;
        float range;
        uint32_t seed;
        double seed_const;
        double range_const;
    };

// There are cases were we want a tensor of all zeros or all ones.
    class UniformTensor : public BaseTensor {
    public:
        UniformTensor(size_t rows, size_t cols, size_t channels, float value) {
            this->rows = rows;
            this->cols = cols;
            this->channels = channels;
            this->value = value;
        }

        void printMaterializationPlan() override {
            cout << "UniformTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return cols;
        }

        size_t channel_count() override {
            return channels;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return value;
        }

    private:
        size_t rows;
        size_t cols;
        size_t channels;
        float value;
    };

    class IdentityTensor : public BaseTensor {
    public:
        IdentityTensor(size_t rows, size_t cols, size_t channels) {
            this->rows = rows;
            this->cols = cols;
            this->channels = channels;
        }

        void printMaterializationPlan() override {
            cout << "IdentityTensor{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t row_count() override {
            return rows;
        }

        size_t column_count() override {
            return cols;
        }

        size_t channel_count() override {
            return channels;
        }

        float get_val(size_t row, size_t column, size_t channel) override {
            return row == column;
        }

    private:
        size_t rows;
        size_t cols;
        size_t channels;
    };

    class BaseTensorUnaryOperatorView : public BaseTensor {
    public:
        explicit BaseTensorUnaryOperatorView(const shared_ptr <BaseTensor> &tensor) {
            this->child = tensor;
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this() || child->contains(other);
        }

        bool readRowsInParallel() override {
            return child->readRowsInParallel();
        }

        size_t row_count() override {
            return child->row_count();
        }

        size_t column_count() override {
            return child->column_count();
        }

        size_t channel_count() override {
            return child->channel_count();
        }

    protected:
        shared_ptr <BaseTensor> child;
    };

// Adds a constant to every value of a matrix through a view
    class TensorAddScalarView : public BaseTensorUnaryOperatorView {
    public:
        TensorAddScalarView(const shared_ptr <BaseTensor> &tensor, float adjustment)
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

    class BaseTensorBinaryOperatorView : public BaseTensor {
    public:
        explicit BaseTensorBinaryOperatorView(const shared_ptr <BaseTensor> &tensor1,
                                              const shared_ptr <BaseTensor> &tensor2) {
            this->child1 = tensor1;
            this->child2 = tensor2;
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this() || child1->contains(other) || child2->contains(other);
        }

        size_t channel_count() override {
            return child1->channel_count();
        }

    protected:
        shared_ptr <BaseTensor> child1;
        shared_ptr <BaseTensor> child2;
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
            if(row < top_padding || row > (child->row_count() + top_padding) ||
               column < left_padding || column > (child->column_count() + left_padding)) {
                return 0.f;
            }

            const float val = child->get_val(row - top_padding, column - left_padding, channel);
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
        TensorValidCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &filter)
                : BaseTensorBinaryOperatorView(tensor, filter) {
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
            const auto filter_rows = child2->row_count();
            const auto filter_cols = child2->column_count();
            float result = 0.f;
            for(size_t current_row = 0; current_row < filter_rows; current_row++) {
                for(size_t current_col = 0; current_col < filter_cols; current_col++) {
                    const auto filter_val = child2->get_val(current_row, current_col, channel);
                    const auto tensor_val = child1->get_val(row+current_row, column+current_col, channel);
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
        TensorFullCrossCorrelation2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &filter)
                : TensorValidCrossCorrelation2dView(
                make_shared<TensorZeroPaddedView>(tensor,
                                                  (size_t)round(((double)filter->row_count() - 1)/2.0),
                                                  (size_t)round(((double)filter->row_count() - 1)/2.0),
                                                  (size_t)round(((double)filter->column_count() - 1)/2.0),
                                                  (size_t)round(((double)filter->column_count() - 1)/2.0)),
                filter) {
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
        TensorFullConvolve2dView(const shared_ptr<BaseTensor> &tensor, const shared_ptr<BaseTensor> &filter)
            : TensorFullCrossCorrelation2dView(tensor, make_shared<TensorRotate180View>(filter)) {

        }

        void printMaterializationPlan() override {
            cout << "TensorFullConvolve2dView{" << row_count() << "," <<column_count()<<","<<channel_count()<<"}->(";
            child1->printMaterializationPlan();
            cout << ") + (";
            child2->printMaterializationPlan();
            cout << ")";
        }
    };

    // channels, rows, columns
    shared_ptr<BaseTensor> materialize_tensor(const shared_ptr<BaseTensor> &other) {
        if(other->isMaterialized()) {
            return other;
        }
        return make_shared<FullTensor>(other);
    }

    shared_ptr<FullTensor> tensor(const vector <vector<vector<float>>> &t) {
        return make_shared<FullTensor>(t);
    }

    shared_ptr<PixelTensor> pixel_tensor(const vector<vector<vector<float>>> &t) {
        return make_shared<PixelTensor>(t);
    }

    shared_ptr<FullTensor> column_vector(const vector<float> &t) {
        return make_shared<FullTensor>(t);
    }

    float scalar(const shared_ptr<BaseTensor> &tensor) {
        if (tensor->size() < 1) {
            return 0.f;
        }
        return tensor->get_val(0);
    }

    shared_ptr<BaseTensor> round(const shared_ptr<BaseTensor> &tensor) {
        return make_shared<TensorRoundedView>(tensor);
    }

    size_t max_index(const shared_ptr<BaseTensor> &tensor) {
        return tensor->max_index(0, 0);
    }


    int estimateBias(int estimate_min, int estimate_max, const float adj_min, const float adj_max) {
        int quarter_bias = estimate_min;
        for(int proposed_quarter_bias = estimate_max; proposed_quarter_bias >= estimate_min; proposed_quarter_bias--) {
            const float bias_max = quarter_to_float(QUARTER_MAX, proposed_quarter_bias);
            const float bias_min = -bias_max;
            if(adj_min > bias_min && adj_max < bias_max) {
                quarter_bias = proposed_quarter_bias;
                break;
            }
        }
        return quarter_bias;
    }

    shared_ptr<BaseTensor> materialize_tensor(const shared_ptr<BaseTensor> &tensor, uint8_t bits) {
        if (bits == 32) {
            if(tensor->isMaterialized()) {
                // there is no advantage to materializing an already materialized tensor to 32 bits.
                // whether other bit options may reduce memory footprint.
                return tensor;
            }
            return make_shared<FullTensor>(tensor);
        } else if (bits == 16) {
            return make_shared<HalfTensor>(tensor);
        }
        auto min_max = tensor->range();
        int quarter_bias = estimateBias(4, 15,  min_max.first, min_max.second);
        return  make_shared<QuarterTensor>(tensor, quarter_bias);
    }
}
#endif //MICROML_TENSOR_HPP
