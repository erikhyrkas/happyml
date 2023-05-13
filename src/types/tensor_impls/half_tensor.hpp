//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_HALF_TENSOR_HPP
#define HAPPYML_HALF_TENSOR_HPP

#include "tensor_allocators.hpp"
#include "../half_float.hpp"
#include <iomanip>
#include <vector>
#include <execution>

namespace happyml {
    class HalfTensor : public BaseAssignableTensor {
    public:
        explicit HalfTensor(const shared_ptr<BaseTensor> &original) {
            const size_t columns = original->columnCount();
            const size_t rows = original->rowCount();
            const size_t channels = original->channelCount();

            allocateTensorVector<uint16_t>(data, rows,
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

        explicit HalfTensor(const vector<float> &values) {
            allocateTensorVector<uint16_t>(data, 1, values.size(), 1);
            size_t col = 0;
            for (float const &val: values) {
                setVal(0, col, 0, val);
                col++;
            }
        }

        explicit HalfTensor(const vector<vector<float>> &values) {
            allocateTensorVector<uint16_t>(data, values.size(), values.at(0).size(), 1);
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
                stream.open(fileName, ifstream::in | ios::binary);
                assignFromStream(stream);
                stream.close();
            } catch (ofstream::failure &e) {
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

            stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            channels = portableBytes(channels);
            stream.read(reinterpret_cast<char *>(&rows), sizeof(rows));
            rows = portableBytes(rows);
            stream.read(reinterpret_cast<char *>(&columns), sizeof(columns));
            columns = portableBytes(columns);

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

        // Don't assign values directly to a tensor. If you have specific values for specific entries,
        // use a view like TensorFromFunction to represent it. Chances are, you don't need to allocate
        // a lot of memory for a full tensor that you will then do other math on. Wait to use memory
        // for the final result.
        inline void setVal(size_t row, size_t column, size_t channel, float val) {
            data.at(channel).at(row).at(column) = floatToHalf(val);
        }
    };
}
#endif //HAPPYML_HALF_TENSOR_HPP
