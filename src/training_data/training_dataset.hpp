//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_TRAINING_DATASET_HPP
#define MICROML_TRAINING_DATASET_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <string>
#include "../types/tensor.hpp"
#include "training_pair.hpp"
#include "data_encoder.hpp"
#include "../util/file_reader.hpp"

namespace microml {

    class TrainingDataSet {
    public:
        virtual size_t record_count() = 0;

        virtual void shuffle() = 0;

        virtual void shuffle(size_t start_offset, size_t end_offset) = 0;

        virtual void restart() = 0;

        virtual std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) = 0;

        virtual std::shared_ptr<TrainingPair> next_record() = 0;

        virtual std::vector<std::vector<size_t>> getGivenShapes() = 0;

        virtual vector<size_t> getGivenShape() {
            return getGivenShapes()[0];
        }

        virtual std::vector<std::vector<size_t>> getExpectedShapes() = 0;

        virtual vector<size_t> getExpectedShape() {
            return getExpectedShapes()[0];
        }
    };

    class EmptyTrainingDataSet : public TrainingDataSet {
        size_t record_count() override {
            return 0;
        }

        void shuffle() override {}

        void shuffle(size_t start_offset, size_t end_offset) override {}

        void restart() override {}

        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            return nullptr;
        }

        std::vector<std::vector<size_t>> getGivenShapes() override {
            return std::vector<std::vector<size_t>>{{1, 1, 1}};
        }

        std::vector<std::vector<size_t>> getExpectedShapes() override {
            return std::vector<std::vector<size_t>>{{1, 1, 1}};
        }
    };

    class PartialTrainingDataSet : public TrainingDataSet {
    public:
        //
        PartialTrainingDataSet(const std::shared_ptr<TrainingDataSet> &dataSource, size_t first_record_offset,
                               size_t last_record_offset) {
            this->dataSource = dataSource;
            this->first_record_offset = first_record_offset;
            this->last_record_offset = last_record_offset;
            this->count = last_record_offset - first_record_offset;
            this->current_offset = first_record_offset;
            if (first_record_offset > last_record_offset) {
                throw std::exception("First offset must be before last offset");
            }
            if (last_record_offset >= dataSource->record_count()) {
                throw std::exception("Record offset out of bounds");
            }
        }

        size_t record_count() override {
            return count;
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            const size_t new_first = first_record_offset + start_offset;
            const size_t new_end = first_record_offset + end_offset;
            if (new_first > last_record_offset || new_end > last_record_offset) {
                throw std::exception("shuffle offset out of range");
            }
            restart();
            dataSource->shuffle(new_first, new_end);
        }

        void shuffle() override {
            shuffle(first_record_offset, last_record_offset);
        }

        void restart() override {
            current_offset = first_record_offset;
        }

        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            const size_t target_last_offset = current_offset + batch_size;
            if (target_last_offset <= last_record_offset) {
                for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                    result.push_back(next_record());
                }
            }
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            std::shared_ptr<TrainingPair> result = dataSource->next_record();
            if (result) {
                current_offset++;
            }
            return result;
        }

    private:
        std::shared_ptr<TrainingDataSet> dataSource;
        size_t first_record_offset;
        size_t last_record_offset;
        size_t count;
        size_t current_offset;
    };


    class InMemoryTrainingDataSet : public TrainingDataSet {
    public:
        InMemoryTrainingDataSet() {
            current_offset = 0;
        }

        void addTrainingData(const shared_ptr<BaseTensor> &given, float expected) {
            pairs.push_back(std::make_shared<TrainingPair>(given, column_vector({expected})));
        }

        void addTrainingData(const shared_ptr<BaseTensor> &given, const shared_ptr<BaseTensor> &expected) {
            pairs.push_back(std::make_shared<TrainingPair>(given, expected));
        }

        size_t record_count() override {
            return pairs.size();
        }

        void shuffle() override {
            // todo: likely can be optimized
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(pairs.begin(), pairs.end(), g);
            restart();
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            // todo: likely can be optimized
            std::random_device rd;
            std::mt19937 g(rd());
            const size_t end = record_count() - end_offset;
            // weird that I had to cast it down to an unsigned long
            // feels like a bug waiting to happen with a large data set.
            std::shuffle(pairs.begin() + (unsigned long) start_offset, pairs.end() - (unsigned long) end, g);
            restart();
        }

        void restart() override {
            current_offset = 0;
        }

        // populate a batch vector of vectors, reusing the structure. This is to save the time we'd otherwise use
        // to allocate.
        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                if (current_offset >= record_count()) {
                    break;
                }
                std::shared_ptr<TrainingPair> next = next_record();
                if (next) {
                    result.push_back(next);
                }
            }
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            if (current_offset >= record_count()) {
                return nullptr;
            }
            std::shared_ptr<TrainingPair> result = pairs.at(current_offset);
            if (result) {
                current_offset++;
            }
            return result;
        }

        std::vector<std::vector<size_t>> getGivenShapes() override {
            if (pairs.empty()) {
                return std::vector<std::vector<size_t>>{{0, 0, 0}};
            }
            auto given = pairs[0]->getGiven();
            std::vector<std::vector<size_t>> result;
            for (const auto &next: given) {
                result.push_back(next->getShape());
            }
            return result;
        }

        std::vector<std::vector<size_t>> getExpectedShapes() override {
            if (pairs.empty()) {
                return std::vector<std::vector<size_t>>{{0, 0, 0}};
            }
            auto expected = pairs[0]->getExpected();
            std::vector<std::vector<size_t>> result;
            for (const auto &next: expected) {
                result.push_back(next->getShape());
            }
            return result;
        }

    private:
        std::vector<std::shared_ptr<TrainingPair>> pairs;
        size_t current_offset;
    };

    // This is fine for small datasets, but
    class InMemoryDelimitedValuesTrainingDataSet : public InMemoryTrainingDataSet {
    public:
        InMemoryDelimitedValuesTrainingDataSet(const string &path, char delimiter,
                                               bool header_row, bool trim_strings, bool expected_first,
                                               size_t expected_columns, size_t given_columns,
                                               const vector<size_t> &expected_shape, const vector<size_t> &given_shape,
                                               const shared_ptr<TrainingDataInputEncoder> &expected_encoder,
                                               const shared_ptr<TrainingDataInputEncoder> &given_encoder)
                : InMemoryTrainingDataSet() {
            this->path = path;
            this->delimiter = delimiter;
            this->header_row = header_row;
            this->trim_strings = trim_strings;
            this->expected_first = expected_first;
            this->expected_columns = expected_columns;
            this->expected_shape = expected_shape;
            this->given_shape = given_shape;
            this->given_columns = given_columns;
            this->expected_encoder = expected_encoder;
            this->given_encoder = given_encoder;
            load();
        }


        std::vector<std::vector<size_t>> getGivenShapes() override {
            return {given_shape};
        }

        std::vector<std::vector<size_t>> getExpectedShapes() override {
            return {expected_shape};
        }

    private:
        string path;
        char delimiter;
        bool expected_first;
        bool header_row;
        bool trim_strings;
        size_t expected_columns;
        size_t given_columns;
        vector<size_t> expected_shape;
        vector<size_t> given_shape;
        shared_ptr<TrainingDataInputEncoder> expected_encoder;
        shared_ptr<TrainingDataInputEncoder> given_encoder;

        void load() {
            DelimitedTextFileReader delimitedTextFileReader(path, delimiter);

            if (header_row && delimitedTextFileReader.hasNext()) {
                delimitedTextFileReader.nextRecord();
            }
            size_t first_size;
            vector<size_t> first_shape;
            vector<size_t> second_shape;
            shared_ptr<TrainingDataInputEncoder> first_encoder;
            shared_ptr<TrainingDataInputEncoder> second_encoder;
            if (expected_first) {
                first_size = expected_columns;
                first_shape = expected_shape;
                first_encoder = expected_encoder;
                second_shape = given_shape;
                second_encoder = given_encoder;
            } else {
                first_size = given_columns;
                first_shape = given_shape;
                first_encoder = given_encoder;
                second_shape = expected_shape;
                second_encoder = expected_encoder;
            }
            while (delimitedTextFileReader.hasNext()) {
                auto record = delimitedTextFileReader.nextRecord();
                auto middleIter(record.begin());
                std::advance(middleIter, first_size);
                std::vector<string> firstHalf(record.begin(), middleIter);
                auto first_tensor = first_encoder->encode(firstHalf, first_shape[0], first_shape[1], first_shape[2],
                                                          trim_strings);
                std::vector<string> secondHalf(middleIter, record.end());
                auto second_tensor = second_encoder->encode(secondHalf, second_shape[0], second_shape[1],
                                                            second_shape[2], trim_strings);
                if (expected_first) {
                    // given, expected
                    addTrainingData(second_tensor, first_tensor);
                } else {
                    addTrainingData(first_tensor, second_tensor);
                }
            }

        }
    };
}

#endif //MICROML_TRAINING_DATASET_HPP
