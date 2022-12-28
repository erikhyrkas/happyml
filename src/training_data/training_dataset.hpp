//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef HAPPYML_TRAINING_DATASET_HPP
#define HAPPYML_TRAINING_DATASET_HPP

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

namespace happyml {

    class TrainingDataSet {
    public:
        virtual size_t recordCount() = 0;

        virtual void shuffle() = 0;

        virtual void shuffle(size_t start_offset, size_t end_offset) = 0;

        virtual void restart() = 0;

        virtual vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) = 0;

        virtual shared_ptr<TrainingPair> nextRecord() = 0;

        virtual vector<vector<size_t>> getGivenShapes() = 0;

        virtual vector<size_t> getGivenShape() {
            return getGivenShapes()[0];
        }

        virtual vector<vector<size_t>> getExpectedShapes() = 0;

        virtual vector<size_t> getExpectedShape() {
            return getExpectedShapes()[0];
        }
    };

    class EmptyTrainingDataSet : public TrainingDataSet {
        size_t recordCount() override {
            return 0;
        }

        void shuffle() override {}

        void shuffle(size_t start_offset, size_t end_offset) override {}

        void restart() override {}

        vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) override {
            vector<shared_ptr<TrainingPair>> result;
            return result;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            return nullptr;
        }

        vector<vector<size_t>> getGivenShapes() override {
            return vector<vector<size_t>>{{1, 1, 1}};
        }

        vector<vector<size_t>> getExpectedShapes() override {
            return vector<vector<size_t>>{{1, 1, 1}};
        }
    };

    class PartialTrainingDataSet : public TrainingDataSet {
    public:
        //
        PartialTrainingDataSet(const shared_ptr<TrainingDataSet> &dataSource, size_t first_record_offset,
                               size_t last_record_offset) {
            this->dataSource = dataSource;
            this->firstRecordOffset = first_record_offset;
            this->lastRecordOffset = last_record_offset;
            this->count = last_record_offset - first_record_offset;
            this->currentOffset = first_record_offset;
            if (first_record_offset > last_record_offset) {
                throw exception("First offset must be before last offset");
            }
            if (last_record_offset >= dataSource->recordCount()) {
                throw exception("Record offset out of bounds");
            }
        }

        size_t recordCount() override {
            return count;
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            const size_t new_first = firstRecordOffset + start_offset;
            const size_t new_end = firstRecordOffset + end_offset;
            if (new_first > lastRecordOffset || new_end > lastRecordOffset) {
                throw exception("shuffle offset out of range");
            }
            restart();
            dataSource->shuffle(new_first, new_end);
        }

        void shuffle() override {
            shuffle(firstRecordOffset, lastRecordOffset);
        }

        void restart() override {
            currentOffset = firstRecordOffset;
        }

        vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) override {
            vector<shared_ptr<TrainingPair>> result;
            const size_t target_last_offset = currentOffset + batch_size;
            if (target_last_offset <= lastRecordOffset) {
                for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                    result.push_back(nextRecord());
                }
            }
            return result;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            shared_ptr<TrainingPair> result = dataSource->nextRecord();
            if (result) {
                currentOffset++;
            }
            return result;
        }

    private:
        shared_ptr<TrainingDataSet> dataSource;
        size_t firstRecordOffset;
        size_t lastRecordOffset;
        size_t count;
        size_t currentOffset;
    };


    class InMemoryTrainingDataSet : public TrainingDataSet {
    public:
        InMemoryTrainingDataSet() {
            current_offset = 0;
        }

        void addTrainingData(const shared_ptr<BaseTensor> &given, float expected) {
            pairs.push_back(make_shared<TrainingPair>(given, columnVector({expected})));
        }

        void addTrainingData(const shared_ptr<BaseTensor> &given, const shared_ptr<BaseTensor> &expected) {
            pairs.push_back(make_shared<TrainingPair>(given, expected));
        }

        size_t recordCount() override {
            return pairs.size();
        }

        void shuffle() override {
            // todo: likely can be optimized
            random_device rd;
            mt19937 g(rd());
            std::shuffle(pairs.begin(), pairs.end(), g);
            restart();
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            // todo: likely can be optimized
            random_device rd;
            mt19937 g(rd());
            const size_t end = recordCount() - end_offset;
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
        vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) override {
            vector<shared_ptr<TrainingPair>> result;
            for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                if (current_offset >= recordCount()) {
                    break;
                }
                shared_ptr<TrainingPair> next = nextRecord();
                if (next) {
                    result.push_back(next);
                }
            }
            return result;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            if (current_offset >= recordCount()) {
                return nullptr;
            }
            shared_ptr<TrainingPair> result = pairs.at(current_offset);
            if (result) {
                current_offset++;
            }
            return result;
        }

        vector<vector<size_t>> getGivenShapes() override {
            if (pairs.empty()) {
                return vector<vector<size_t>>{{0, 0, 0}};
            }
            auto given = pairs[0]->getGiven();
            vector<vector<size_t>> result;
            for (const auto &next: given) {
                result.push_back(next->getShape());
            }
            return result;
        }

        vector<vector<size_t>> getExpectedShapes() override {
            if (pairs.empty()) {
                return vector<vector<size_t>>{{0, 0, 0}};
            }
            auto expected = pairs[0]->getExpected();
            vector<vector<size_t>> result;
            for (const auto &next: expected) {
                result.push_back(next->getShape());
            }
            return result;
        }

    private:
        vector<shared_ptr<TrainingPair>> pairs;
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
            this->headerRow = header_row;
            this->trimStrings = trim_strings;
            this->expectedFirst = expected_first;
            this->expectedColumns = expected_columns;
            this->expectedShape = expected_shape;
            this->givenShape = given_shape;
            this->givenColumns = given_columns;
            this->expectedEncoder = expected_encoder;
            this->givenEncoder = given_encoder;
            load();
        }


        vector<vector<size_t>> getGivenShapes() override {
            return {givenShape};
        }

        vector<vector<size_t>> getExpectedShapes() override {
            return {expectedShape};
        }

    private:
        string path;
        char delimiter;
        bool expectedFirst;
        bool headerRow;
        bool trimStrings;
        size_t expectedColumns;
        size_t givenColumns;
        vector<size_t> expectedShape;
        vector<size_t> givenShape;
        shared_ptr<TrainingDataInputEncoder> expectedEncoder;
        shared_ptr<TrainingDataInputEncoder> givenEncoder;

        void load() {
            DelimitedTextFileReader delimitedTextFileReader(path, delimiter);

            if (headerRow && delimitedTextFileReader.hasNext()) {
                delimitedTextFileReader.nextRecord();
            }
            size_t first_size;
            vector<size_t> first_shape;
            vector<size_t> second_shape;
            shared_ptr<TrainingDataInputEncoder> firstEncoder;
            shared_ptr<TrainingDataInputEncoder> secondEncoder;
            if (expectedFirst) {
                first_size = expectedColumns;
                first_shape = expectedShape;
                firstEncoder = expectedEncoder;
                second_shape = givenShape;
                secondEncoder = givenEncoder;
            } else {
                first_size = givenColumns;
                first_shape = givenShape;
                firstEncoder = givenEncoder;
                second_shape = expectedShape;
                secondEncoder = expectedEncoder;
            }
            while (delimitedTextFileReader.hasNext()) {
                auto record = delimitedTextFileReader.nextRecord();
                auto middleIter(record.begin());
                std::advance(middleIter, first_size);
                vector<string> firstHalf(record.begin(), middleIter);
                auto firstTensor = firstEncoder->encode(firstHalf, first_shape[0], first_shape[1], first_shape[2],
                                                        trimStrings);
                vector<string> secondHalf(middleIter, record.end());
                auto secondTensor = secondEncoder->encode(secondHalf, second_shape[0], second_shape[1],
                                                          second_shape[2], trimStrings);
                if (expectedFirst) {
                    // given, expected
                    addTrainingData(secondTensor, firstTensor);
                } else {
                    addTrainingData(firstTensor, secondTensor);
                }
            }

        }
    };
}

#endif //HAPPYML_TRAINING_DATASET_HPP
