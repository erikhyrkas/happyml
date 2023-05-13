//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_PIXEL_TENSOR_HPP
#define HAPPYML_PIXEL_TENSOR_HPP

#include "tensor_allocators.hpp"
#include <iomanip>
#include <vector>
#include <utility>
#include <execution>

namespace happyml {
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
            const size_t columns = original->columnCount();
            const size_t rows = original->rowCount();
            const size_t channels = original->channelCount();

            allocateTensorVector<uint8_t>(data, rows,
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

        // If you use this constructor, you've already wasted a lot of memory.
        // Maybe you can just use a full tensor?
        explicit PixelTensor(const vector<float> &values) {
            allocateTensorVector<uint8_t>(data, 1, values.size(), 1);
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
            allocateTensorVector<uint8_t>(data, values[0].size(), values[0][0].size(), values.size());
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
                stream.open(fileName, ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch (ofstream::failure &e) {
                cerr << "Failed to load: " << fileName << endl << e.what() << endl;
                throw e;
            }
        }

        explicit PixelTensor(ifstream &stream) {
            assignFromStream(stream);
        }

        explicit PixelTensor(ifstream &stream, uint64_t rows, uint64_t columns, uint64_t channels) {
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
            return ((float) data.at(channel).at(row).at(column)) / 255.f;
        }

        void printMaterializationPlan() override {
            cout << "PixelTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
        }

    private:
        vector<vector<vector<uint8_t>>> data;

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
            data.at(channel).at(row).at(column) = (uint8_t) (std::max(0.0f, std::min(val, 1.0f)) * 255);
        }
    };
}
#endif //HAPPYML_PIXEL_TENSOR_HPP
