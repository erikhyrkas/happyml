//
// Created by Erik Hyrkas on 11/2/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TRAINING_DATASET_HPP
#define HAPPYML_TRAINING_DATASET_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <string>
#include <unordered_set>
#include "../types/tensor.hpp"
#include "training_pair.hpp"
#include "data_encoder.hpp"
#include "../util/file_reader.hpp"

namespace happyml {

    class TrainingDataSet {
    public:
        virtual size_t recordCount() = 0;

        virtual void shuffle() = 0;

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

//
//    class PagingDataSet : public TrainingDataSet {
//    public:
//        PagingDataSet(const std::string& file_path, char delimiter, bool skip_header, bool trim_strings,
//                      const shared_ptr<DataEncoder>& given_encoder, const shared_ptr<DataEncoder>& expected_encoder,
//                      const vector<size_t>& given_shape, const vector<size_t>& expected_shape)
//                : current_offset(0), delimiter(delimiter), skip_header(skip_header), trimStrings(trim_strings),
//                  given_encoder(given_encoder), expected_encoder(expected_encoder),
//                  given_shape(given_shape), expected_shape(expected_shape) {
//            training_data_path = file_path;
//            total_records = countRecords(file_path);
//        }
//
//        size_t recordCount() override {
//            return total_records;
//        }
//
//        void shuffle() override {
//            // TODO: implement shuffle method
////            random_device rd;
////            mt19937 g(rd());
////            std::shuffle(pairs.begin(), pairs.end(), g);
//            restart();
//        }
//
//        void restart() override {
//            current_offset = 0;
//        }
//
//        vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) override {
//            if (current_offset >= recordCount()) {
//                return vector<shared_ptr<TrainingPair>>{};
//            }
//
//            auto batch = loadBatch(batch_size);
//            current_offset += batch_size;
//            return batch;
//        }
//
//    private:
//        size_t current_offset;
//        size_t total_records = 0;
//        string training_data_path;
//        char delimiter;
//        bool skip_header;
//        bool trimStrings;
//
//        shared_ptr<DataEncoder> given_encoder;
//        shared_ptr<DataEncoder> expected_encoder;
//        vector<size_t> given_shape;
//        vector<size_t> expected_shape;
//
//        [[nodiscard]] size_t countRecords(const std::string& file_path) const {
//            size_t record_count = 0;
//            DelimitedTextFileReader reader(file_path, delimiter, skip_header);
//            while (reader.hasNext()) {
//                reader.nextRecord();
//                record_count++;
//            }
//            return record_count;
//        }
//
//        vector<shared_ptr<TrainingPair>> loadBatch(size_t batch_size) {
//            vector<shared_ptr<TrainingPair>> batch;
//            DelimitedTextFileReader reader(training_data_path, delimiter, skip_header);
//            size_t counter = 0;
//
//            while (reader.hasNext() && counter < current_offset + batch_size) {
//                auto record = reader.nextRecord();
//                if (counter >= current_offset) {
//                    auto middleIter(record.begin());
//                    std::advance(middleIter, given_shape[1]);
//                    vector<string> given_part(record.begin(), middleIter);
//                    auto given_tensor = given_encoder->encode(given_part, given_shape[0], given_shape[1], given_shape[2],
//                                                              trimStrings);
//
//                    vector<string> expected_part(middleIter, record.end());
//                    auto expected_tensor = expected_encoder->encode(expected_part, expected_shape[0], expected_shape[1], expected_shape[2],
//                                                                    trimStrings);
//
//                    batch.push_back(make_shared<TrainingPair>(given_tensor, expected_tensor));
//                }
//                counter++;
//            }
//
//            return batch;
//        }
//    };

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

        void restart() override {
            current_offset = 0;
        }

        // Set of the distinct values for a given column, useful for finding labels.
        unordered_set<float> getDistinctValuesForFirstExpectedColumn(size_t columnIndex) {
            unordered_set<float> distinctValues;
            for (const auto &pair : pairs) {
                const auto &expectedTensor = pair->getFirstExpected();
                if (columnIndex < expectedTensor->columnCount()) {
                    float value = expectedTensor->getValue(0, columnIndex, 0);
                    distinctValues.insert(value);
                }
            }
            return distinctValues;
        }

        // Set of the distinct values for a given column, useful for finding labels.
        unordered_set<float> getDistinctValuesForFirstGivenColumn(size_t columnIndex) {
            unordered_set<float> distinctValues;
            for (const auto &pair : pairs) {
                const auto &expectedTensor = pair->getFirstGiven();
                if (columnIndex < expectedTensor->columnCount()) {
                    float value = expectedTensor->getValue(0, columnIndex, 0);
                    distinctValues.insert(value);
                }
            }
            return distinctValues;
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

    // This is fine for small datasets, but won't work at scale because not all datasets will fit in memory
    class InMemoryDelimitedValuesTrainingDataSet : public InMemoryTrainingDataSet {
    public:
        InMemoryDelimitedValuesTrainingDataSet(const string &path, char delimiter,
                                               bool header_row, bool trim_strings, bool expected_first,
                                               size_t expected_columns, size_t given_columns,
                                               const vector<size_t> &expected_shape, const vector<size_t> &given_shape,
                                               const shared_ptr<DataEncoder> &expected_encoder,
                                               const shared_ptr<DataEncoder> &given_encoder)
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
        shared_ptr<DataEncoder> expectedEncoder;
        shared_ptr<DataEncoder> givenEncoder;

        void load() {
            DelimitedTextFileReader delimitedTextFileReader(path, delimiter, headerRow);

            size_t first_size;
            vector<size_t> first_shape;
            vector<size_t> second_shape;
            shared_ptr<DataEncoder> firstEncoder;
            shared_ptr<DataEncoder> secondEncoder;
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
