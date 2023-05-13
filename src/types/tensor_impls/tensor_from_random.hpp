//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_FROM_RANDOM_HPP
#define HAPPYML_TENSOR_FROM_RANDOM_HPP

namespace happyml {
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
}

#endif //HAPPYML_TENSOR_FROM_RANDOM_HPP
