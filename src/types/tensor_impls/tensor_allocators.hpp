//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_ALLOCATORS_HPP
#define HAPPYML_TENSOR_ALLOCATORS_HPP

#include <vector>
#include <future>
#include <execution>

namespace happyml {
    template<typename T>
    void allocateTensorVector(vector<vector<vector<T>>> &data, const size_t rows, const size_t columns,
                              const size_t channels) {
        // I wouldn't expect memory allocation in parallel to be faster than serial allocation, especially on this
        // scale, since there would be a lot of lock contention. I did a quick experiment and found the timings
        // weren't better or worse with small sizes. More experiments are needed, but it's not worth much time
        // at this moment since I'm skeptical it would help.
        data.resize(channels);
        // #pragma omp for
        for (size_t channel = 0; channel < channels; channel++) {
            data.at(channel).resize(rows);
            // omp for is roughly 5-10% slower here.
            // #pragma omp for
            for (size_t row = 0; row < rows; row++) {
                data.at(channel).at(row).resize(columns);
            }
        }
    }

// This is roughly 10% slower than the copy-and-pasted code. That's a huge hit. I'm leaving
// this here, in case I can think of a way to improve on this.
// Example usage:
//            allocateTensorVector<float>(data, rows,
//                                        columns,
//                                        channels,
//                                        original,
//                                        [](float original) {return original;});
    template<typename T>
    void allocateTensorVector(vector<vector<vector<T>>> &data,
                              const size_t rows, const size_t columns, const size_t channels,
                              const shared_ptr<happyml::BaseTensor> &original,
                              function<T(float)> conversionFunction) {
        allocateTensorVector<T>(data, rows,
                                columns,
                                channels);

//#pragma omp for collapse(3)
        for (long long channel = 0; channel < channels; channel++) {
            for (long long row = 0; row < rows; row++) {
                for (long long column = 0; column < columns; column++) {
                    data.at(channel).at(row).at(column) = conversionFunction(original->getValue(row, column, channel));
                }
            }
        }
    }
}
#endif //HAPPYML_TENSOR_ALLOCATORS_HPP
