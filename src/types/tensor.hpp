//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_HPP
#define HAPPYML_TENSOR_HPP

#include <execution>
#include <future>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <fstream>
#include <set>
#include "quarter_float.hpp"
#include "half_float.hpp"
#include "../util/portable_bytes.hpp"
#include "../util/index_value.hpp"

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

namespace happyml {

    class BaseTensor : public enable_shared_from_this<BaseTensor> {
    public:
        virtual size_t rowCount() = 0;

        virtual size_t columnCount() = 0;

        virtual size_t channelCount() = 0;

        virtual bool isMaterialized() {
            return false;
        }

        void save(ofstream &stream, bool no_header = false) {
            uint64_t channels = channelCount();
            uint64_t rows = rowCount();
            uint64_t columns = columnCount();
            if(no_header) {
                auto portableChannels = portableBytes(channels);
                stream.write(reinterpret_cast<const char *>(&portableChannels), sizeof(portableChannels));
                auto portableRows = portableBytes(rows);
                stream.write(reinterpret_cast<const char *>(&portableRows), sizeof(portableRows));
                auto portableColumns = portableBytes(columns);
                stream.write(reinterpret_cast<const char *>(&portableColumns), sizeof(portableColumns));
            }

            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t column = 0; column < columns; column++) {
                        float floatVal = getValue(row, column, channel);
                        uint32_t portableVal = portableBytes(*(uint32_t *) &floatVal);
                        stream.write(reinterpret_cast<const char *>(&portableVal), sizeof(portableVal));
                    }
                }
            }
        }

        bool save(const string &fileName) {
            try {
                ofstream stream;
                stream.open(fileName, std::ofstream::out | ios::binary | ios::trunc);
                save(stream);
                stream.close();
                return true;
            } catch (ofstream::failure &e) {
                // I was torn about catching an exception and returning true/false
                // this is inconsistent with the load method.
                cerr << "Failed to save: " << fileName << endl << e.what() << endl;
                return false;
            }
        }

        virtual void printMaterializationPlanLine() {
            printMaterializationPlan();
            cout << endl;
        }

        virtual void printMaterializationPlan() = 0;

        // fastest read is generally along columns because of how memory is organized,
        // but we can't do a parallel read if there's only one row.
        virtual bool readRowsInParallel() {
            return (rowCount() > 1);
        }

        virtual bool contains(const shared_ptr<BaseTensor> &other) {
            return other == shared_from_this();
        }

        unsigned long size() {
            return rowCount() * columnCount() * channelCount();
        }

        unsigned long elementsPerChannel() {
            return rowCount() * columnCount();
        }

        virtual float getValue(size_t row, size_t column, size_t channel) = 0;

        virtual vector <size_t> getShape() {
            return {rowCount(), columnCount(), channelCount()};
        }

        vector<float> getRowValues(size_t row, size_t channel=0) {
            vector<float> nextRow;
            size_t columnCount1 = columnCount();
            for(size_t col = 0; col < columnCount1; col++) {
                nextRow.push_back(getValue(row, col, 0));
            }
            return nextRow;
        }

        float getValue(const unsigned long position_offset) {
            const size_t cols = columnCount();
            const unsigned long matrix_size = cols * rowCount();
            const size_t new_channel = position_offset / matrix_size;
            const unsigned long matrix_elements = position_offset % matrix_size;
            const size_t new_row = matrix_elements / cols;
            const size_t new_col = matrix_elements % cols;
            return getValue(new_row, new_col, new_channel);
        }

        double product() {
            double result = 1.0;
            const size_t maxRows = rowCount();
            const size_t maxCols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < maxRows; row++) {
                    for (size_t col = 0; col < maxCols; col++) {
                        result *= getValue(row, col, channel);
                    }
                }
            }
            return result;
        }

        double sum() {
            double result = 0.0;
            const size_t maxRows = rowCount();
            const size_t maxCols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < maxRows; row++) {
                    for (size_t col = 0; col < maxCols; col++) {
                        result += getValue(row, col, channel);
                    }
                }
            }
            return result;
        }

        float max() {
            float result = -INFINITY;
            const size_t maxRows = rowCount();
            const size_t maxCols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < maxRows; row++) {
                    for (size_t col = 0; col < maxCols; col++) {
                        result = std::max(result, getValue(row, col, channel));
                    }
                }
            }
            return result;
        }

        float min() {
            float result = INFINITY;
            const size_t maxRows = rowCount();
            const size_t maxCols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < maxRows; row++) {
                    for (size_t col = 0; col < maxCols; col++) {
                        result = std::min(result, getValue(row, col, channel));
                    }
                }
            }
            return result;
        }

        pair<float, float> range() {
            float minResult = INFINITY;
            float maxResult = -INFINITY;
            const size_t maxRows = rowCount();
            const size_t maxCols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < maxRows; row++) {
                    for (size_t col = 0; col < maxCols; col++) {
                        const auto val = getValue(row, col, channel);
                        minResult = std::min(minResult, val);
                        maxResult = std::max(maxResult, val);
                    }
                }
            }
            return {minResult, maxResult};
        }

        vector <IndexValue> topIndices(size_t numberOfResults, size_t channel, size_t row) {
            vector<IndexValue> result;
            if (numberOfResults > 0) {
                multiset<IndexValue> sortedValues;
                const size_t maxCols = columnCount();
                for (size_t col = 0; col < maxCols; col++) {
                    float nextVal = getValue(row, col, channel);
                    sortedValues.insert(IndexValue(col, nextVal));
                }
                multiset<IndexValue>::reverse_iterator reverseIterator;
                for (reverseIterator = sortedValues.rbegin();
                     reverseIterator != sortedValues.rend(); reverseIterator++) {
                    result.push_back(*reverseIterator);
                    if (result.size() == numberOfResults) {
                        break;
                    }
                }
            }
            return result;
        }

        size_t maxIndex(size_t channel, size_t row) {
            size_t result = 0;
            float currentMax = -INFINITY;
            const size_t maxCols = columnCount();
            for (size_t col = 0; col < maxCols; col++) {
                float nextVal = getValue(row, col, channel);
                if (nextVal > currentMax) {
                    currentMax = nextVal;
                    result = col;
                }
            }
            return result;
        }

        size_t minIndex(size_t channel, size_t row) {
            size_t result = 0;
            float currentMin = INFINITY;
            const size_t maxCols = columnCount();
            for (size_t col = 0; col < maxCols; col++) {
                float nextVal = getValue(row, col, channel);
                if (nextVal < currentMin) {
                    currentMin = nextVal;
                    result = col;
                }
            }
            return result;
        }

        vector <size_t> maxIndices(size_t channel, size_t row) {
            vector<size_t> result;
            float current_max = -INFINITY;
            const size_t maxCols = columnCount();
            for (size_t col = 0; col < maxCols; col++) {
                float nextVal = getValue(row, col, channel);
                if (nextVal > current_max) {
                    current_max = nextVal;
                    result.clear();
                    result.push_back(col);
                } else if (nextVal == current_max) {
                    result.push_back(col);
                }
            }
            return result;
        }

        vector <size_t> minIndices(size_t channel, size_t row) {
            vector<size_t> result;
            float currentMin = INFINITY;
            const size_t maxCols = columnCount();
            for (size_t col = 0; col < maxCols; col++) {
                float nextVal = getValue(row, col, channel);
                if (nextVal < currentMin) {
                    currentMin = nextVal;
                    result.clear();
                    result.push_back(col);
                } else if (nextVal == currentMin) {
                    result.push_back(col);
                }
            }
            return result;
        }

        float arithmeticMean() {
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
            const size_t rows = rowCount();
            const size_t cols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < cols; col++) {
                        index++;
                        const double val = getValue(row, col, channel);
                        average += (val - average) / index;
                    }
                }
            }

            return (float) average;
        }

        float geometricMean() {
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
            const size_t rows = rowCount();
            const size_t cols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < cols; col++) {
                        const double val = ((double) getValue(row, col, channel));
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
            out << setprecision(3) << fixed << endl;
            const size_t rows = rowCount();
            const size_t cols = columnCount();
            const size_t maxChannels = channelCount();
            for (size_t channel = 0; channel < maxChannels; channel++) {
                if (maxChannels > 1) {
                    out << "[" << endl;
                }
                for (size_t row = 0; row < rows; row++) {
                    if (rows > 1) {
                        cout << "|";
                    } else {
                        cout << "[";
                    }
                    string delim;
                    for (size_t col = 0; col < cols; col++) {
                        out << delim << getValue(row, col, channel);
                        delim = ", ";
                    }
                    if (rows > 1) {
                        out << "|" << endl;
                    } else {
                        out << "]" << endl;
                    }
                }
                if (maxChannels > 1) {
                    out << "]" << endl;
                }
            }
        }
    };

// This abstract class lets us build float tensors and bit tensors as well and use them interchangeably.
    class BaseAssignableTensor : public BaseTensor {
    public:
        bool isMaterialized() override {
            return true;
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
            cout << "TensorFromFunction{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return cols;
        }

        size_t channelCount() override {
            return channels;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
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
                TensorFromRandom(rows, cols, channels, quarterToFloat(QUARTER_MIN, bias),
                                 quarterToFloat(QUARTER_MAX, bias), 42) {
        }

        TensorFromRandom(size_t rows, size_t cols, size_t channels, int bias, uint32_t seed) :
                TensorFromRandom(rows, cols, channels, quarterToFloat(QUARTER_MIN, bias),
                                 quarterToFloat(QUARTER_MAX, bias), seed) {
        }

        void printMaterializationPlan() override {
            cout << "TensorFromRandom{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return cols;
        }

        size_t channelCount() override {
            return channels;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
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
            cout << "UniformTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return cols;
        }

        size_t channelCount() override {
            return channels;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
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
            cout << "IdentityTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this();
        }

        size_t rowCount() override {
            return rows;
        }

        size_t columnCount() override {
            return cols;
        }

        size_t channelCount() override {
            return channels;
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return row == column;
        }

    private:
        size_t rows;
        size_t cols;
        size_t channels;
    };

    class BaseTensorUnaryOperatorView : public BaseTensor {
    public:
        explicit BaseTensorUnaryOperatorView(const shared_ptr<BaseTensor> &tensor) {
            this->child = tensor;
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this() || child->contains(other);
        }

        bool readRowsInParallel() override {
            return child->readRowsInParallel();
        }

        size_t rowCount() override {
            return child->rowCount();
        }

        size_t columnCount() override {
            return child->columnCount();
        }

        size_t channelCount() override {
            return child->channelCount();
        }

    protected:
        shared_ptr<BaseTensor> child;
    };


    class BaseTensorBinaryOperatorView : public BaseTensor {
    public:
        explicit BaseTensorBinaryOperatorView(const shared_ptr<BaseTensor> &tensor1,
                                              const shared_ptr<BaseTensor> &tensor2) {
            this->child1 = tensor1;
            this->child2 = tensor2;
        }

        bool contains(const shared_ptr<BaseTensor> &other) override {
            return other == shared_from_this() || child1->contains(other) || child2->contains(other);
        }

        size_t channelCount() override {
            return child1->channelCount();
        }

    protected:
        shared_ptr<BaseTensor> child1;
        shared_ptr<BaseTensor> child2;
    };


}
#endif //HAPPYML_TENSOR_HPP
