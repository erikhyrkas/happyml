//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_UNIFORM_TENSOR_HPP
#define HAPPYML_UNIFORM_TENSOR_HPP

namespace happyml {
// There are cases were we want a tensor of all zeros or all ones.
    class UniformTensor : public BaseTensor {
    public:
        UniformTensor(const vector<size_t> &shape, float value) {
            this->rows = shape[0];
            this->cols = shape[1];
            this->channels = shape[2];
            this->value = value;
        }

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
}

#endif //HAPPYML_UNIFORM_TENSOR_HPP
