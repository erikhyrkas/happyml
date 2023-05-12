//
// Created by Erik Hyrkas on 5/12/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_FROM_XAVIER_HPP
#define HAPPYML_TENSOR_FROM_XAVIER_HPP

namespace happyml {

    class TensorFromXavier : public happyml::BaseTensor {
    public:
        TensorFromXavier(size_t rows, size_t cols, size_t channels, uint32_t seed) {
            this->rows = rows;
            this->cols = cols;
            this->channel_size = (double) rows * (double) cols;
            this->channels = channels;
            this->seed = seed;

            // Xavier/Glorot initialization
            float variance = sqrtf(2.0f / (float)(rows + cols))/2.0f;
            this->min_value = -variance;
            this->max_value = variance;
            this->range = fabs(max_value - min_value);
            this->range_const = range / 2.71828;
            this->seed_const = (std::min(seed, (uint32_t) 1) * range_const) / 3.14159265358979323846;
        }


        void printMaterializationPlan() override {
            cout << "TensorFromXavier{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
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
#endif //HAPPYML_TENSOR_FROM_XAVIER_HPP
