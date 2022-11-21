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
void wait_for_futures(std::queue<std::future<void>> &futures) {
    while (!futures.empty()) {
        futures.front().wait();
        futures.pop();
    }
}

class BaseTensor {
public:
    virtual size_t row_count() = 0;

    virtual size_t column_count() = 0;

    virtual size_t channel_count() = 0;

    virtual bool contains(BaseTensor *other) {
        return other == this;
    }

    unsigned long size() {
        return row_count() * column_count() * channel_count();
    }

    unsigned long elements_per_channel() {
        return row_count() * column_count();
    }

    virtual float get_val(size_t row, size_t column, size_t channel) = 0;

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
        //        double scale_val = std::max(std::abs(max()), std::abs(min()))*10.0;
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
        print(std::cout);
    }

    void print(std::ostream &out) {
        out << "[" << std::setprecision(3) << std::fixed << std::endl;
        const size_t rows = row_count();
        const size_t cols = column_count();
        const size_t max_channels = channel_count();
        for (size_t channel = 0; channel < max_channels; channel++) {
            out << "[" << std::endl;
            for (size_t row = 0; row < rows; row++) {
                std::string delim;
                for (size_t col = 0; col < cols; col++) {
                    out << delim << get_val(row, col, channel);
                    delim = ", ";
                }
                out << std::endl;
            }
            out << "]" << std::endl;
        }
        out << "]" << std::endl;
    }
};

// This abstract class lets us build float tensors and bit tensors as well and use them interchangeably.
class BaseAssignableTensor : public BaseTensor {
public:
    virtual void assign(BaseTensor &other) = 0;

    virtual void assign(BaseTensor &other, BaseAssignableTensor &working_memory) = 0;
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

    explicit FullTensor(BaseTensor &original)
            : FullTensor(original.row_count(),
                         original.column_count(),
                         original.channel_count()) {
        do_assign(original);
    }

    explicit FullTensor(const std::vector<float> &values) : FullTensor(1, values.size(), 1) {
        size_t col = 0;
        for (float const &val: values) {
            set_val(0, col, 0, val);
            col++;
        }
    }

    void assign(BaseTensor &other, BaseAssignableTensor &working_memory) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy to working memory -- needs tests" << std::endl;
            working_memory.assign(other);
            do_assign(working_memory);
        } else {
            std::cout << "ASSIGN: no copy to working memory -- needs tests" << std::endl;
            do_assign(other);
        }
    }

    // will assign the values from other to this tensor, but if the other tensor is
    // a view that contains us, then we avoid data corruption by copying to a temporary
    // tensor first.
    // this could possibly be optimized by leveraging the newly allocated tensor's
    // internal values directly, but our current data elements are vectors, not a pointer
    // to vectors.
    void assign(BaseTensor &other) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy -- needs tests" << std::endl;
            auto temp = std::make_shared<FullTensor>(other);
//            do_assign(*temp); -- let's just steal the memory:
            data = std::move(temp->data);
        } else {
            std::cout << "ASSIGN: no copy -- needs tests" << std::endl;
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

private:
    std::vector<std::vector<std::vector<float>>> data;

    void do_assign(BaseTensor &other) {
        if (other.row_count() != row_count() && other.channel_count() != channel_count() &&
            other.column_count() != column_count()) {
            throw std::exception(
                    "A tensor cannot be assigned from another tensor with a different shape");
        }

        const size_t columns = column_count();
        const size_t rows = row_count();
        const size_t channels = channel_count();
        if (rows <= 1 && columns < 10000) {
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    populate_by_col(&other, this, row, columns, channel);
                }
            }
        } else {
            std::queue<std::future<void>> futures;
            const size_t wait_amount = 8096;
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    auto next_async = std::async(std::launch::async, populate_by_col, &other, this, row, columns,
                                                 channel);
                    futures.push(std::move(next_async));
                    if (futures.size() >= wait_amount) {
                        wait_for_futures(futures);
                    }
                }
            }
            wait_for_futures(futures);
        }
    }


    static void populate_by_col(BaseTensor *source,
                                FullTensor *dest,
                                size_t row,
                                size_t max_cols,
                                size_t channel) {
        for (size_t col = 0; col < max_cols; col++) {
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

    explicit PixelTensor(BaseTensor &original)
            : PixelTensor(original.row_count(),
                         original.column_count(),
                         original.channel_count()) {
        do_assign(original);
    }

    explicit PixelTensor(const std::vector<float> &values) : PixelTensor(1, values.size(), 1) {
        size_t col = 0;
        for (float const &val: values) {
            set_val(0, col, 0, val);
            col++;
        }
    }

    void assign(BaseTensor &other, BaseAssignableTensor &working_memory) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy to working memory -- needs tests" << std::endl;
            working_memory.assign(other);
            do_assign(working_memory);
        } else {
            std::cout << "ASSIGN: no copy to working memory -- needs tests" << std::endl;
            do_assign(other);
        }
    }

    // will assign the values from other to this tensor, but if the other tensor is
    // a view that contains us, then we avoid data corruption by copying to a temporary
    // tensor first.
    // this could possibly be optimized by leveraging the newly allocated tensor's
    // internal values directly, but our current data elements are vectors, not a pointer
    // to vectors.
    void assign(BaseTensor &other) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy -- needs tests" << std::endl;
            auto temp = std::make_shared<PixelTensor>(other);
//            do_assign(*temp); -- let's just steal the memory:
            data = std::move(temp->data);
        } else {
            std::cout << "ASSIGN: no copy -- needs tests" << std::endl;
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

private:
    std::vector<std::vector<std::vector<uint8_t>>> data;

    void do_assign(BaseTensor &other) {
        if (other.row_count() != row_count() && other.channel_count() != channel_count() &&
            other.column_count() != column_count()) {
            throw std::exception(
                    "A tensor cannot be assigned from another tensor with a different shape");
        }

        const size_t columns = column_count();
        const size_t rows = row_count();
        const size_t channels = channel_count();
        if (rows <= 1 && columns < 10000) {
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    populate_by_col(&other, this, row, columns, channel);
                }
            }
        } else {
            std::queue<std::future<void>> futures;
            const size_t wait_amount = 8096;
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    auto next_async = std::async(std::launch::async, populate_by_col, &other, this, row, columns,
                                                 channel);
                    futures.push(std::move(next_async));
                    if (futures.size() >= wait_amount) {
                        wait_for_futures(futures);
                    }
                }
            }
            wait_for_futures(futures);
        }
    }


    static void populate_by_col(BaseTensor *source,
                                PixelTensor *dest,
                                size_t row,
                                size_t max_cols,
                                size_t channel) {
        for (size_t col = 0; col < max_cols; col++) {
            dest->set_val(row, col, channel, source->get_val(row, col, channel));
        }
    }

    inline void set_val(size_t row, size_t column, size_t channel, float val) {
        data.at(channel).at(row).at(column) = (uint8_t)(std::max(0.0f, std::min(val, 1.0f)) * 255);
    }
};

class QuarterTensor : public BaseAssignableTensor {
public:
    QuarterTensor(const size_t rows, const size_t columns, const size_t channels, const int bias, const float offset) {
        this->bias = bias;
        this->offset = offset;
        data.resize(channels);
        for (size_t channel = 0; channel < channels; channel++) {
            data.at(channel).resize(rows);
            for (size_t row = 0; row < rows; row++) {
                data.at(channel).at(row).resize(columns);
            }
        }
    }

    explicit QuarterTensor(BaseTensor &original, const int bias, const float offset)
            : QuarterTensor(original.row_count(),
                            original.column_count(),
                            original.channel_count(),
                            bias,
                            offset) {
        do_assign(original);
    }

    QuarterTensor(const std::vector<float> &values, const int bias, const float offset) : QuarterTensor(1,
                                                                                                        values.size(),
                                                                                                        1,
                                                                                                        bias,
                                                                                                        offset) {
        size_t col = 0;
        for (float const &val: values) {
            set_val(0, col, 0, val);
            col++;
        }
    }

    QuarterTensor(const std::vector<std::vector<float>> &values, const int bias, const float offset) : QuarterTensor(
            values.size(),
            values.at(0).size(),
            1,
            bias,
            offset) {
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
        return quarter_to_float(data.at(channel).at(row).at(column), bias, offset);
    }


    [[nodiscard]] int get_bias() const {
        return bias;
    }

    [[nodiscard]] float get_offset() const {
        return offset;
    }

    // We're potentially dealing with huge amounts of memory, and we don't want to
    // allocate and reallocate. If the tensor we are assigning from contains ourselves
    // we'd corrupt the data if we wrote to the tensor while reading from it, so we
    // need working memory to hold the temporary results.
    void assign(BaseTensor &other, BaseAssignableTensor &working_memory) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy to working memory -- needs tests" << std::endl;
            working_memory.assign(other);
            do_assign(working_memory);
        } else {
            std::cout << "ASSIGN: no copy to working memory -- needs tests" << std::endl;
            do_assign(other);
        }
    }

    // will assign the values from other to this tensor, but if the other tensor is
    // a view that contains us, then we avoid data corruption by copying to a temporary
    // tensor first.
    // this could possibly be optimized by leveraging the newly allocated tensor's
    // internal values directly, but our current data elements are vectors, not a pointer
    // to vectors.
    void assign(BaseTensor &other) override {
        if (&other == this) {
            return; //assignment to self is pointless and expensive
        }
        if (contains(&other)) {
            std::cout << "ASSIGN: making copy -- needs tests" << std::endl;
            auto temp = std::make_shared<QuarterTensor>(other, bias, offset);
//            do_assign(*temp); -- let's just steal the memory:
            data = std::move(temp->data);
        } else {
            std::cout << "ASSIGN: no copy -- needs tests" << std::endl;
            do_assign(other);
        }
    }

private:
    std::vector<std::vector<std::vector<quarter>>> data;
    int bias;
    float offset;

    void do_assign(BaseTensor &other) {
        if (other.row_count() != row_count() && other.channel_count() != channel_count() &&
            other.column_count() != column_count()) {
            throw std::exception(
                    "A tensor cannot be assigned from another tensor with a different shape");
        }

        const size_t columns = column_count();
        const size_t rows = row_count();
        const size_t channels = channel_count();
        if (rows <= 1 && columns < 10000) {
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    populate_by_col(&other, this, row, columns, channel);
                }
            }
        } else {
            std::queue<std::future<void>> futures;
            const size_t wait_amount = 8096;
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    auto next_async = std::async(std::launch::async, populate_by_col, &other, this, row, columns,
                                                 channel);
                    futures.push(std::move(next_async));
                    if (futures.size() >= wait_amount) {
                        wait_for_futures(futures);
                    }
                }
            }
            wait_for_futures(futures);
        }
    }


    static void populate_by_col(BaseTensor *source,
                                QuarterTensor *dest,
                                size_t row,
                                size_t max_cols,
                                size_t channel) {
        for (size_t col = 0; col < max_cols; col++) {
            dest->set_val(row, col, channel, source->get_val(row, col, channel));
        }
    }

    // Don't assign values directly to a tensor. If you have specific values for specific entries,
    // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
    // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
    // for the final result.
    inline void set_val(size_t row, size_t column, size_t channel, float val) {
        data.at(channel).at(row).at(column) = float_to_quarter(val, bias, offset);
    }
};

//class RowVector : public QuarterTensor {
//public:
//    RowVector(const size_t columns, const int bias, const float offset, const bool zero_out) : QuarterTensor(1, columns,
//                                                                                                             1,
//                                                                                                             bias,
//                                                                                                             offset,
//                                                                                                             zero_out) {}
//
//    RowVector(BaseTensor &original, const size_t columns) : QuarterTensor(original, 1, columns, 1) {}
//};
//
//class ColumnVector : public QuarterTensor {
//public:
//    ColumnVector(const size_t rows, const int bias, const float offset, const bool zero_out) : QuarterTensor(rows, 1, 1,
//                                                                                                             bias,
//                                                                                                             offset,
//                                                                                                             zero_out) {}
//
//    ColumnVector(BaseTensor &original, const size_t rows) : QuarterTensor(original, rows, 1, 1) {}
//};

// If you can represent a tensor as a function, we don't have to allocate gigabytes of memory
// to hold it. You already have a compact representation of it.
class TensorFromFunction : public BaseTensor {
public:
    TensorFromFunction(std::function<float(size_t, size_t, size_t)> tensorFunction, size_t rows, size_t cols,
                       size_t channels) {
        this->tensorFunction = std::move(tensorFunction);
        this->rows = rows;
        this->cols = cols;
        this->channels = channels;
    }

    bool contains(BaseTensor *other) override {
        return other == this;
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
    std::function<float(size_t, size_t, size_t)> tensorFunction;
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
        this->range = std::fabs(max_value - min_value);
        this->range_const = range / 2.71828;
        this->seed_const = (std::min(seed, (uint32_t) 1) * range_const) / 3.14159265358979323846;
    }

    TensorFromRandom(size_t rows, size_t cols, size_t channels, int bias) :
            TensorFromRandom(rows, cols, channels, quarter_to_float(QUARTER_MIN, bias, 0),
                             quarter_to_float(QUARTER_MAX, bias, 0), 42) {
    }

    TensorFromRandom(size_t rows, size_t cols, size_t channels, int bias, uint32_t seed) :
            TensorFromRandom(rows, cols, channels, quarter_to_float(QUARTER_MIN, bias, 0),
                             quarter_to_float(QUARTER_MAX, bias, 0), seed) {
    }


    bool contains(BaseTensor *other) override {
        return other == this;
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
        return (float) (max_value - std::fmod(offset, range));
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

    bool contains(BaseTensor *other) override {
        return other == this;
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

    bool contains(BaseTensor *other) override {
        return other == this;
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
    explicit BaseTensorUnaryOperatorView(const std::shared_ptr<BaseTensor> &tensor) {
        this->child = tensor;
    }

    bool contains(BaseTensor *other) override {
        return other == this || child->contains(other);
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
    std::shared_ptr<BaseTensor> child;
};

// Adds a constant to every value of a matrix through a view
class TensorAddScalarView : public BaseTensorUnaryOperatorView {
public:
    TensorAddScalarView(const std::shared_ptr<BaseTensor> &tensor, float adjustment)
            : BaseTensorUnaryOperatorView(tensor) {
        this->adjustment = adjustment;
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
    TensorMultiplyByScalarView(const std::shared_ptr<BaseTensor> &tensor, float scale) : BaseTensorUnaryOperatorView(
            tensor) {
        this->scale = scale;
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
    TensorValueTransformView(const std::shared_ptr<BaseTensor> &tensor, std::function<float(float)> transformFunction)
            : BaseTensorUnaryOperatorView(
            tensor) {
        this->transformFunction = std::move(transformFunction);
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        return transformFunction(child->get_val(row, column, channel));
    }

private:
    std::function<float(float)> transformFunction;
};

class TensorValueTransform2View : public BaseTensorUnaryOperatorView {
public:
    TensorValueTransform2View(const std::shared_ptr<BaseTensor> &tensor,
                              std::function<float(float, std::vector<double>)> transformFunction,
                              std::vector<double> constants) : BaseTensorUnaryOperatorView(
            tensor) {
        this->transformFunction = std::move(transformFunction);
        this->constants = std::move(constants);
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        return transformFunction(child->get_val(row, column, channel), constants);
    }

private:
    std::function<float(float, std::vector<double>)> transformFunction;
    std::vector<double> constants;
};

// Change the number of rows and columns, but maintain the same number of elements per channel.
// You cannot change the number of channels in the current implementation.
class TensorReshapeView : public BaseTensorUnaryOperatorView {
public:
    TensorReshapeView(const std::shared_ptr<BaseTensor> &tensor, const size_t rows,
                      const size_t columns) : BaseTensorUnaryOperatorView(tensor) {
        this->rows = rows;
        this->columns = columns;
        this->elements_per_channel = (unsigned long) rows * (unsigned long) columns;
        if (tensor->elements_per_channel() != elements_per_channel) {
            throw std::exception("A matrix view must be put over a matrix with the same number of elements.");
        }
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
    TensorFlattenToRowView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        this->columns = tensor->size();
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

    float get_val(size_t row, size_t column, size_t channel) override {
        if(row != 0 || channel != 0) {
            throw std::exception("Row Vector has only a single row and channel.");
        }
        return child->get_val(column);
    }


private:
    size_t columns;
};

// Converts a 3d tensor into a column vector
class TensorFlattenToColumnView : public BaseTensorUnaryOperatorView {
public:
    TensorFlattenToColumnView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
        this->rows = tensor->size();
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

    float get_val(size_t row, size_t column, size_t channel) override {
        if(column != 0 || channel != 0) {
            throw std::exception("Column Vector has only a single column and channel.");
        }
        return child->get_val(row);
    }


private:
    size_t rows;
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
    TensorDiagonalView(const std::shared_ptr<BaseTensor> &tensor, size_t row_offset) : BaseTensorUnaryOperatorView(
            tensor) {
        this->row_offset = row_offset;
        // we only have as many columns as there were rows
        this->columns = tensor->row_count() - row_offset;
        // we either have 0 or 1 result row
        this->rows = row_offset < tensor->row_count();
    }

    explicit TensorDiagonalView(const std::shared_ptr<BaseTensor> &tensor)
            : TensorDiagonalView(tensor, 0) {
    }

    size_t row_count() override {
        return rows;
    }

    size_t column_count() override {
        return columns;
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        // we aren't bounds checking, so the caller better make sure that row_count > 0
        return child->get_val(column + row_offset, column, channel);
    }

private:
    size_t row_offset;
    size_t columns;
    size_t rows;
};


class TensorNoOpView : public BaseTensorUnaryOperatorView {
public:
    explicit TensorNoOpView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {}

    float get_val(size_t row, size_t column, size_t channel) override {
        return child->get_val(row, column, channel);
    }

private:
};

class BaseTensorBinaryOperatorView : public BaseTensor {
public:
    explicit BaseTensorBinaryOperatorView(const std::shared_ptr<BaseTensor> &tensor1,
                                          const std::shared_ptr<BaseTensor> &tensor2) {
        this->child1 = tensor1;
        this->child2 = tensor2;
    }

    bool contains(BaseTensor *other) override {
        return other == this || child1->contains(other) || child2->contains(other);
    }

    size_t channel_count() override {
        return child1->channel_count();
    }

protected:
    std::shared_ptr<BaseTensor> child1;
    std::shared_ptr<BaseTensor> child2;
};

class TensorDotTensorView : public BaseTensorBinaryOperatorView {
public:
    TensorDotTensorView(const std::shared_ptr<BaseTensor> &tensor1,
                        const std::shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
        if (tensor1->column_count() != tensor2->row_count()) {
            throw std::exception("Dot product tensor1.cols must match tensor2.rows in length");
        }
        if (tensor1->channel_count() != tensor2->channel_count()) {
            throw std::exception("Dot product tensor1.channels must match tensor2.channels in length");
        }
    }

    size_t row_count() override {
        return child1->row_count();
    }

    size_t column_count() override {
        return child2->column_count();
    }

    float get_val(size_t row, size_t column, size_t channel) override {
//        std::cout << "getting val: " << row << ", " << column << std::endl;
        float val = 0;
        for (size_t t1_col = 0; t1_col < child1->column_count(); t1_col++) {
//            std::cout << "... + "<< child1->get_val(row, t1_col, channel) <<" (" << row << ", " << t1_col << ") * " << child2->get_val(t1_col, column, channel) << "( " << t1_col << ", " << column << ")" << std::endl;
            val += child1->get_val(row, t1_col, channel) * child2->get_val(t1_col, column, channel);
        }
        return val;
    }
};


class TensorAddTensorView : public BaseTensorBinaryOperatorView {
public:
    TensorAddTensorView(const std::shared_ptr<BaseTensor> &tensor1,
                        const std::shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
        if (tensor1->channel_count() != tensor2->channel_count() || tensor1->row_count() != tensor2->row_count() ||
            tensor1->column_count() != tensor2->column_count()) {
            throw std::exception("You can only add two tensors of the same dimensions together.");
        }
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
    TensorMinusTensorView(const std::shared_ptr<BaseTensor> &tensor1,
                          const std::shared_ptr<BaseTensor> &tensor2) : BaseTensorBinaryOperatorView(tensor1, tensor2) {
        if (tensor1->channel_count() != tensor2->channel_count() || tensor1->row_count() != tensor2->row_count() ||
            tensor1->column_count() != tensor2->column_count()) {
            throw std::exception("You can only add two tensors of the same dimensions together.");
        }
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
    TensorPowerView(const std::shared_ptr<BaseTensor> &tensor, const float power) : BaseTensorUnaryOperatorView(
            tensor) {
        this->power = power;
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        const float val = child->get_val(row, column, channel);
        return powf(val, power);
    }

private:
    float power;
};

class TensorLogView : public BaseTensorUnaryOperatorView {
public:
    explicit TensorLogView(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        const float val = child->get_val(row, column, channel);
        return log(val);
    }

private:
};

class TensorLog2View : public BaseTensorUnaryOperatorView {
public:
    explicit TensorLog2View(const std::shared_ptr<BaseTensor> &tensor) : BaseTensorUnaryOperatorView(tensor) {
    }

    float get_val(size_t row, size_t column, size_t channel) override {
        const float val = child->get_val(row, column, channel);
        return log2(val);
    }

private:
};

#endif //MICROML_TENSOR_HPP
