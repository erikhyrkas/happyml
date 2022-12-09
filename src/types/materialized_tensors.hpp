//
// Created by Erik Hyrkas on 12/9/2022.
//

#ifndef MICROML_MATERIALIZED_TENSORS_HPP
#define MICROML_MATERIALIZED_TENSORS_HPP
#include <execution>
#include <future>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include "quarter_float.hpp"
#include "half_float.hpp"
#include "tensor.hpp"

using namespace std;

namespace microml {
    void wait_for_futures(queue <future<void>> &futures) {
        while (!futures.empty()) {
            futures.front().wait();
            futures.pop();
        }
    }

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
                // TODO: I think I should switch to #pragma omp for collapse(3)
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
                // TODO: I think I should switch to #pragma omp for collapse(3)
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
                // TODO: I think I should switch to #pragma omp for collapse(3)
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
                // TODO: I think I should switch to #pragma omp for collapse(3)

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
}
#endif //MICROML_MATERIALIZED_TENSORS_HPP
