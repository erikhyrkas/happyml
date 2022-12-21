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
#include "../util/portable_bytes.hpp"

using namespace std;

// NOTE: There's a lot of duplicate code here. I could likely reduce it through the use of templates,
//  which I may do. My biggest hesitation right now is the setVal function, which I would like to remain inlined
//  for performance. getVal and setVal both are responsible for data conversion through function calls of their own.
//  Because we are dealing with base types that aren't classes, I'm not sure how to use a templated class
//  to handle conversion neatly without losing performance. It's probably possible and something I need to learn,
//  since the below code has SO MUCH DUPLICATION.
namespace microml {
// The full tensor is backed by a 32-bit float. This exists because our input into our models may
// require accurate representations, and I don't think they'll ever be too big to fit in memory.
// There may also be final dense layers that have few enough neurons feeding it that a full tensor
// may work.
    class FullTensor : public BaseAssignableTensor {
    public:
        explicit FullTensor(const shared_ptr<BaseTensor> &original) {
            allocate(original->rowCount(),
                     original->columnCount(),
                     original->channelCount());

            const size_t columns = columnCount();
            const size_t rows = rowCount();
            const size_t channels = channelCount();

            #pragma omp for collapse(3)
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < columns; col++) {
                        setVal(row, col, channel, original->getValue(row, col, channel));
                    }
                }
            }
        }

        explicit FullTensor(const vector<float> &values) {
            allocate(1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        // get a weird warning here that CLion can't resolve constructor. I believe this is a bug with CLion itself:
        // https://youtrack.jetbrains.com/issue/CPP-24510/Bad-detection-of-Constructor-is-not-implemented
        explicit FullTensor(const vector<vector<vector<float>>> &values) {
            allocate(values[0].size(), values[0][0].size(), values.size());
            size_t channel_index = 0;
            for (const vector <vector<float>> &next_channel: values) {
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
                stream.open(fileName,ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch(ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit FullTensor(ifstream &stream) {
            assignFromStream(stream);
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
        vector <vector<vector < float>>> data;

        void allocate(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        void assignFromStream(ifstream &stream) {
            uint64_t channels;
            uint64_t rows;
            uint64_t columns;

            stream.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
            columns = portableBytes(columns);

            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                    for(size_t column = 0; column < columns; column++) {
                        uint32_t val;
                        stream.read(reinterpret_cast<char*>(&val), sizeof(val));
                        val = portableBytes(val);
                        float nextVal = *(float*) &val;
                        setVal(row, column, channel, nextVal);
                    }
                }
            }
        }

        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = val;
        }
    };


// TODO: Okay, so I clearly need another layer of abstraction or to template the BaseAssignableTensor, but
//  that'll be another day.
// Pixel Tensor holds a value between 0.0f and 1.0f with an even distribution in 256 increments (8-bits.)
// This is a compact representation useful for images, but also for other data that has an evenly distributed
// range of values between 0 and 1 with a similar granularity.
// The quarter tensor with a bias of 14 is capable of a similar representation, but the distribution of values isn't
// even. This Tensor is also faster than the quarter tensor because far less math needs to happen to map between
// float and 8-bits.
    class PixelTensor : public BaseAssignableTensor {
    public:

        explicit PixelTensor(const shared_ptr<BaseTensor> &original) {
            allocate(original->rowCount(),
                     original->columnCount(),
                     original->channelCount());

            const size_t columns = columnCount();
            const size_t rows = rowCount();
            const size_t channels = channelCount();

            #pragma omp for collapse(3)
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < columns; col++) {
                        setVal(row, col, channel, original->getValue(row, col, channel));
                    }
                }
            }
        }

        // If you use this constructor, you've already wasted a lot of memory.
        // Maybe you can just use a full tensor?
        explicit PixelTensor(const vector<float> &values) {
            allocate(1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        // see the note by FullTensor about the CLion warning bug.
        // If you use this constructor, you've already wasted a lot of memory.
        // Maybe you can just use a full tensor?
        explicit PixelTensor(const vector<vector<vector<float>>> &values) {
            allocate(values[0].size(), values[0][0].size(), values.size());
            size_t channel_index = 0;
            for (const auto &next_channel: values) {
                size_t row_index = 0;
                for (const auto &next_row: next_channel) {
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

        explicit PixelTensor(const string &fileName) {
            try {
                ifstream stream;
                stream.open(fileName,ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch(ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit PixelTensor(ifstream &stream) {
            assignFromStream(stream);
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
            return ((float) data.at(channel).at(row).at(column)) / 255.f;
        }
        void printMaterializationPlan() override {
            cout << "PixelTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }
    private:
        vector<vector<vector<uint8_t>>> data;

        void allocate(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        void assignFromStream(ifstream &stream) {
            uint64_t channels;
            uint64_t rows;
            uint64_t columns;

            stream.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
            columns = portableBytes(columns);

            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                    for(size_t column = 0; column < columns; column++) {
                        uint32_t val;
                        stream.read(reinterpret_cast<char*>(&val), sizeof(val));
                        val = portableBytes(val);
                        float nextVal = *(float*) &val;
                        setVal(row, column, channel, nextVal);
                    }
                }
            }
        }

        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = (uint8_t) (std::max(0.0f, std::min(val, 1.0f)) * 255);
        }
    };

    class QuarterTensor : public BaseAssignableTensor {
    public:
        explicit QuarterTensor(const shared_ptr<BaseTensor> &original, const int bias) {
            this->bias = bias;
            allocate(original->rowCount(),
                     original->columnCount(),
                     original->channelCount());

            const size_t columns = columnCount();
            const size_t rows = rowCount();
            const size_t channels = channelCount();

            #pragma omp for collapse(3)
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < columns; col++) {
                        setVal(row, col, channel, original->getValue(row, col, channel));
                    }
                }
            }
        }

        QuarterTensor(const vector<float> &values, const int bias) {
            this->bias = bias;
            allocate(1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        QuarterTensor(const vector <vector<float>> &values, const int bias) {
            this->bias = bias;
            allocate( values.size(), values.at(0).size(), 1);
            for (size_t row = 0; row < values.size(); row++) {
                for (size_t col = 0; col < values[row].size(); col++) {
                    const float val = values.at(row).at(col);
                    setVal(row, col, 0, val);
                }
            }
        }

        explicit QuarterTensor(const string &fileName, const int bias) {
            this->bias = bias;
            try {
                ifstream stream;
                stream.open(fileName,ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch(ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit QuarterTensor(ifstream &stream, const int bias) {
            this->bias = bias;
            assignFromStream(stream);
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
            return quarterToFloat(data.at(channel).at(row).at(column), bias);
        }


        [[nodiscard]] int get_bias() const {
            return bias;
        }


        void printMaterializationPlan() override {
            cout << "QuarterTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }
    private:
        vector<vector<vector<quarter>>> data;
        int bias;

        void allocate(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        void assignFromStream(ifstream &stream) {
            uint64_t channels;
            uint64_t rows;
            uint64_t columns;

            stream.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
            columns = portableBytes(columns);

            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                    for(size_t column = 0; column < columns; column++) {
                        uint32_t val;
                        stream.read(reinterpret_cast<char*>(&val), sizeof(val));
                        val = portableBytes(val);
                        float nextVal = *(float*) &val;
                        setVal(row, column, channel, nextVal);
                    }
                }
            }
        }

        // Don't assign values directly to a tensor. If you have specific values for specific entries,
        // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
        // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
        // for the final result.
        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = floatToQuarter(val, bias);
        }
    };


    class HalfTensor : public BaseAssignableTensor {
    public:
        explicit HalfTensor(const shared_ptr<BaseTensor> &original) {
            allocate(original->rowCount(),
                     original->columnCount(),
                     original->channelCount());

            const size_t columns = columnCount();
            const size_t rows = rowCount();
            const size_t channels = channelCount();
            #pragma omp for collapse(3)
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < columns; col++) {
                        setVal(row, col, channel, original->getValue(row, col, channel));
                    }
                }
            }
        }

        explicit HalfTensor(const vector<float> &values) {
            allocate(1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        explicit HalfTensor(const vector<vector<float>> &values) {
            allocate(values.size(), values.at(0).size(), 1);
            for (size_t row = 0; row < values.size(); row++) {
                for (size_t col = 0; col < values[row].size(); col++) {
                    const float val = values.at(row).at(col);
                    setVal(row, col, 0, val);
                }
            }
        }

        explicit HalfTensor(const string &fileName) {
            try {
                ifstream stream;
                stream.open(fileName,ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch(ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit HalfTensor(ifstream &stream) {
            assignFromStream(stream);
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
            return halfToFloat(data.at(channel).at(row).at(column));
        }

        void printMaterializationPlan() override {
            cout << "HalfTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

    private:
        vector<vector<vector<half>>> data;

        void assignFromStream(ifstream &stream) {
            uint64_t channels;
            uint64_t rows;
            uint64_t columns;

            stream.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
            columns = portableBytes(columns);

            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                    for(size_t column = 0; column < columns; column++) {
                        uint32_t val;
                        stream.read(reinterpret_cast<char*>(&val), sizeof(val));
                        val = portableBytes(val);
                        float nextVal = *(float*) &val;
                        setVal(row, column, channel, nextVal);
                    }
                }
            }
        }

        void allocate(const size_t rows, const size_t columns, const size_t channels) {
            data.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                data.at(channel).resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    data.at(channel).at(row).resize(columns);
                }
            }
        }

        // Don't assign values directly to a tensor. If you have specific values for specific entries,
        // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
        // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
        // for the final result.
        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = floatToHalf(val);
        }
    };
}
#endif //MICROML_MATERIALIZED_TENSORS_HPP
