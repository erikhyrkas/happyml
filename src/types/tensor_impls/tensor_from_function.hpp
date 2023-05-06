//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_FROM_FUNCTION_HPP
#define HAPPYML_TENSOR_FROM_FUNCTION_HPP

namespace happyml {
// If you can represent a tensor as a function, we don't have to allocate gigabytes of memory
// to hold it. You already have a compact representation of it.
    class TensorFromFunction : public happyml::BaseTensor {
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
}

#endif //HAPPYML_TENSOR_FROM_FUNCTION_HPP
