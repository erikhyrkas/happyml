//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_IDENTITY_TENSOR_HPP
#define HAPPYML_IDENTITY_TENSOR_HPP

namespace happyml {
    class IdentityTensor : public happyml::BaseTensor {
    public:
        IdentityTensor(size_t rows, size_t cols, size_t channels) {
            this->rows = rows;
            this->cols = cols;
            this->channels = channels;
        }

        void printMaterializationPlan() override {
            cout << "IdentityTensor{" << rowCount() << "," << columnCount() << "," << channelCount() << "}";
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
            return row == column;
        }

    private:
        size_t rows;
        size_t cols;
        size_t channels;
    };
}
#endif //HAPPYML_IDENTITY_TENSOR_HPP
