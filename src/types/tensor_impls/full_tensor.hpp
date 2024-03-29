//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FULL_TENSOR_HPP
#define HAPPYML_FULL_TENSOR_HPP

#include <execution>
#include <vector>
#include <iomanip>
#include "tensor_allocators.hpp"

using namespace std;

// NOTE: There's a lot of duplicate code here. I could likely reduce it through the use of templates,
//  which I may do. My biggest hesitation right now is the setVal function, which I would like to remain inlined
//  for performance. getVal and setVal both are responsible for data conversion through function calls of their own.
//  Because we are dealing with base types that aren't classes, I'm not sure how to use a templated class
//  to handle conversion neatly without losing performance. It's probably possible and something I need to learn,
//  since the below code has SO MUCH DUPLICATION.
// UPDATED THOUGHT: maybe the way to achieve DRY (Don't Repeat Yourself) would be to make template-able inlinable
//  functions that do operations on the vectors. These functions wouldn't be part of a specific class, instead
//  they would just be standard functions that multiple classes could use. There would be some amount of repetition
//  but far less than there is today.
namespace happyml {
    // TODO: create a "solid-state" tensor, where instead of being in memory, use disk and mem-map it as needed.
    //  My c++ is rusty, so I need to dig into how this is done at some point. I know it is possible, but there
    //  may be crazy OS-specific requirements or other such nonsense to work through that isn't important to tackle
    //  at this exact moment.

    // TODO: is there a way to create a tensor backed by GPU memory? my whole approach is driven by optimizing for
    //  CPU and regular memory. I mean, I can think of ways of putting the tensors in GPU memory, but operations
    //  are currently applied one tensor entry at a time, not the entire tensor, which isn't how GPU operations
    //  need to work. I think that I'd need to build GPU specific tensors for every view and the 32-bit and 16-bit
    //  materialized tensors. I'd probably have to wrap the construction of those tensors and views with a function
    //  that builds the correct version. Sounds like a fair amount of work, but it might be the only way. I'll
    //  think on this more. This approach would be less GPU memory efficient than other frameworks that don't
    //  have immutable tensors. They update the original when it makes sense and create new tensors when it makes
    //  sense. This framework used views for all of the intermediate steps and only persisted if you made one of the
    //  explicit materialized tensors.
    //  ADDITIONAL THOUGHT: even if the views had a method for applying a GPU operation equivalent, a
    //  materialization method would not know if it could update the base tensors and would need to create new
    //  GPU-based tensors and then apply the views that were GPU specific. The end result being a fair amount of
    //  GPU memory usage.
    //  YET ANOTHER THOUGHT: The only way I can see to safely do this is to have all tensors made through factory
    //  methods and those factory methods would need to have a parameter that states whether GPU memory was required
    //  and in those cases, it would always return a materialized tensor, so you never had a stack of views for
    //  a GPU-based model. Every single tensor was materialized when it was made. CPU-based tensors would work as they
    //  do today where they are a few materialized tensors with a stack of views on top of them, but GPU-based
    //  would never be a view -- the operation you performed on them would immediate return a new GPU-based tensor.
    //  We could possibly take in another flag that let the caller specify whether the original tensor needed to be
    //  retained or if it could be updated. This would allow us to reuse GPU-memory in some cases, but it would never
    //  be as efficient as other ML frameworks in the exact same situations. We might get close, though.
    //  DISCARDED THOUGHT: I considered if there was a way through pointer reference counts to decide if
    //  the original tensor could be overwritten, but I don't think this would be safe or reliable. I think the
    //  explicit instruction to reuse memory is still the best option.


    // TODO: create a "sparse tensor". There are cases where inputs are mostly 0s (or some other value.) There's no need
    //  to use a large amount of memory to hold the same value. Have a default value the tensor returns for a row/column
    //  and only return a different value if it is specified. This would be slower than a FullTensor, but useful
    //  when we want to be memory efficient.

    // The full tensor is backed by a 32-bit float. This exists because our input into our models may
// require accurate representations, and I don't think they'll ever be too big to fit in memory.
// There may also be final dense layers that have few enough neurons feeding it that a full tensor
// may work.
    class FullTensor : public BaseAssignableTensor {
    public:
        explicit FullTensor(const shared_ptr<BaseTensor> &original) {
            const size_t columns = original->columnCount();
            const size_t rows = original->rowCount();
            const size_t channels = original->channelCount();

            allocateTensorVector<float>(data, rows,
                                        columns,
                                        channels);

//#pragma omp for collapse(3)
            for (long long channel = 0; channel < channels; channel++) {
                for (long long row = 0; row < rows; row++) {
                    for (long long col = 0; col < columns; col++) {
                        setVal(row, col, channel, original->getValue(row, col, channel));
                    }
                }
            }
        }

        explicit FullTensor(const vector<float> &values) {
            allocateTensorVector<float>(data, 1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        // get a weird warning here that CLion can't resolve constructor. I believe this is a bug with CLion itself:
        // https://youtrack.jetbrains.com/issue/CPP-24510/Bad-detection-of-Constructor-is-not-implemented
        explicit FullTensor(const vector<vector<vector<float>>> &values) {
            allocateTensorVector<float>(data, values[0].size(), values[0][0].size(), values.size());
            // TODO: this could be made parallel, but we'd have to index rather than use iterator
            size_t channel_index = 0;
            for (const vector<vector<float>> &next_channel: values) {
                size_t row_index = 0;
                for (const vector<float> &next_row: next_channel) {
                    size_t col_index = 0;
                    for (float val: next_row) {
                        setVal(row_index, col_index, channel_index, val);
                        col_index++;
                    }
                    row_index++;
                }
                channel_index++;
            }
        }

        explicit FullTensor(const string &fileName) {
            try {
                ifstream stream;
                stream.open(fileName, ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch (ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit FullTensor(ifstream &stream) {
            assignFromStream(stream);
        }

        explicit FullTensor(ifstream &stream, uint64_t rows, uint64_t columns, uint64_t channels) {
            assignBodyFromStream(stream, rows, columns, channels);
        }

        size_t channelCount() override {
            return data.size();
        }

        size_t rowCount() override {
            if (data.empty()) {
                return 0;
            }
            return data[0].size();
        }

        size_t columnCount() override {
            if (data.empty() || data[0].empty()) {
                return 0;
            }
            return data[0][0].size();
        }

        float getValue(size_t row, size_t column, size_t channel) override {
            return data.at(channel).at(row).at(column);
        }

        void printMaterializationPlan() override {
            cout << "FullTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

    private:
        vector<vector<vector<float>>> data;

        void assignFromStream(ifstream &stream) {
            uint64_t channels;
            uint64_t rows;
            uint64_t columns;

            stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char *>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char *>(&columns), sizeof(columns));
            columns = portableBytes(columns);

            assignBodyFromStream(stream, rows, columns, channels);
        }

        void assignBodyFromStream(ifstream &stream, uint64_t rows, uint64_t columns, uint64_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                    for (size_t column = 0; column < columns; column++) {
                        uint32_t val;
                        stream.read(reinterpret_cast<char *>(&val), sizeof(val));
                        val = portableBytes(val);
                        float nextVal = *(float *) &val;
                        setVal(row, column, channel, nextVal);
                    }
                }
            }
        }

        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = val;
        }
    };


}
#endif //HAPPYML_FULL_TENSOR_HPP
