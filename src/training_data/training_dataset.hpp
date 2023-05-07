//
// Created by Erik Hyrkas on 11/2/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TRAINING_DATASET_HPP
#define HAPPYML_TRAINING_DATASET_HPP

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include "training_pair.hpp"
#include "../util/shuffler.hpp"
#include "../util/file_reader.hpp"
#include "../util/file_writer.hpp"

namespace happyml {

    class TrainingDataSet {
    public:
        virtual size_t recordCount() = 0;

        virtual void restart() = 0;

        virtual shared_ptr<TrainingPair> nextRecord() = 0;

        virtual vector<vector<size_t>> getGivenShapes() = 0;

        virtual vector<size_t> getGivenShape() {
            return getGivenShapes()[0];
        }

        virtual vector<vector<size_t>> getExpectedShapes() = 0;

        virtual vector<size_t> getExpectedShape() {
            return getExpectedShapes()[0];
        }

        void setShuffler(const shared_ptr<Shuffler> &shuffler) {
            if (shuffler != nullptr && shuffler->getSize() != recordCount()) {
                throw runtime_error("Shuffler needs to be sized appropriately for the dataset.");
            }
            shuffler_ = shuffler;
        }

    protected:
        shared_ptr<Shuffler> shuffler_;
    };

    class EmptyTrainingDataSet : public TrainingDataSet {
        size_t recordCount() override {
            return 0;
        }

        void restart() override {}

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
            if (shuffler_ != nullptr) {
                throw runtime_error("Cannot add data after a shuffler has been assigned");
            }
            pairs.push_back(make_shared<TrainingPair>(given, columnVector({expected})));
        }

        void addTrainingData(const shared_ptr<BaseTensor> &given, const shared_ptr<BaseTensor> &expected) {
            if (shuffler_ != nullptr) {
                throw runtime_error("Cannot add data after a shuffler has been assigned");
            }
            pairs.push_back(make_shared<TrainingPair>(given, expected));
        }

        void addTrainingData(const vector<shared_ptr<BaseTensor>> &given, const vector<shared_ptr<BaseTensor>> &expected) {
            if (shuffler_ != nullptr) {
                throw runtime_error("Cannot add data after a shuffler has been assigned");
            }
            pairs.push_back(make_shared<TrainingPair>(given, expected));
        }

        size_t recordCount() override {
            return pairs.size();
        }

        void restart() override {
            current_offset = 0;
        }

//        // Set of the distinct values for a given column, useful for finding labels.
//        unordered_set<float> getDistinctValuesForFirstExpectedColumn(size_t columnIndex) {
//            unordered_set<float> distinctValues;
//            for (const auto &pair: pairs) {
//                const auto &expectedTensor = pair->getFirstExpected();
//                if (columnIndex < expectedTensor->columnCount()) {
//                    float value = expectedTensor->getValue(0, columnIndex, 0);
//                    distinctValues.insert(value);
//                }
//            }
//            return distinctValues;
//        }
//
//        // Set of the distinct values for a given column, useful for finding labels.
//        unordered_set<float> getDistinctValuesForFirstGivenColumn(size_t columnIndex) {
//            unordered_set<float> distinctValues;
//            for (const auto &pair: pairs) {
//                const auto &expectedTensor = pair->getFirstGiven();
//                if (columnIndex < expectedTensor->columnCount()) {
//                    float value = expectedTensor->getValue(0, columnIndex, 0);
//                    distinctValues.insert(value);
//                }
//            }
//            return distinctValues;
//        }

        shared_ptr<TrainingPair> nextRecord() override {
            if (current_offset >= recordCount()) {
                return nullptr;
            }
            auto shuffled_offset = shuffler_ != nullptr ? shuffler_->getShuffledIndex(current_offset) : current_offset;
            shared_ptr<TrainingPair> result = pairs.at(shuffled_offset);
            if (result) {
                current_offset++;
            }
            return result;
        }

        vector<vector<size_t>> getGivenShapes() override {
            if (pairs.empty()) {
                return vector<vector<size_t>>{{0, 0, 0}};
            }
            if (givenShape.empty()) {
                auto given = pairs[0]->getGiven();
                givenShape.reserve(given.size());
                for (const auto &next: given) {
                    givenShape.push_back(next->getShape());
                }

            }
            return givenShape;
        }

        vector<vector<size_t>> getExpectedShapes() override {
            if (pairs.empty()) {
                return vector<vector<size_t>>{{0, 0, 0}};
            }
            if (expectedShape.empty()) {
                auto expected = pairs[0]->getExpected();
                expectedShape.reserve(expected.size());
                for (const auto &next: expected) {
                    expectedShape.push_back(next->getShape());
                }
            }
            return expectedShape;
        }

    private:
        vector<shared_ptr<TrainingPair>> pairs;
        size_t current_offset;
        vector<vector<size_t>> expectedShape;
        vector<vector<size_t>> givenShape;
    };

    struct ColumnGroup {
        ColumnGroup() : id_(0), start_index(0), source_column_count(0), rows(0), columns(0), channels(0) {}

        ColumnGroup(size_t id, size_t startIndex, size_t source_column_count, string use, string dataType, size_t rows, size_t columns, size_t channels) :
                id_(id), start_index(startIndex), source_column_count(source_column_count), use(std::move(use)), data_type(std::move(dataType)), rows(rows), columns(columns), channels(channels) {}

        explicit ColumnGroup(const shared_ptr<ColumnGroup> &from) {
            id_ = from->id_;
            start_index = from->start_index;
            source_column_count = from->source_column_count;
            use = from->use;
            data_type = from->data_type;
            rows = from->rows;
            columns = from->columns;
            channels = from->channels;
            ordered_distinct_labels_ = from->ordered_distinct_labels_;
        }

        ColumnGroup(const shared_ptr<ColumnGroup> &from, vector<string> ordered_distinct_labels) {
            id_ = from->id_;
            start_index = from->start_index;
            source_column_count = from->source_column_count;
            use = from->use;
            data_type = from->data_type;
            rows = from->rows;
            columns = from->columns;
            channels = from->channels;
            ordered_distinct_labels_ = std::move(ordered_distinct_labels);
        }

        size_t start_index;
        size_t source_column_count;
        string use; // given or expected
        string data_type; // image, label, number, text
        size_t rows;
        size_t columns;
        size_t channels;
        size_t id_;
        vector<string> ordered_distinct_labels_;
    };

    bool update_column_positions(const string &original_file,
                                 const string &new_file,
                                 const vector<shared_ptr<ColumnGroup>> &given_columns,
                                 const vector<shared_ptr<ColumnGroup>> &expected_columns,
                                 bool has_header) {
        DelimitedTextFileReader reader(original_file, ',', has_header);
        DelimitedTextFileWriter writer(new_file, ',');
        size_t records_updated = 0;
        while (reader.hasNext()) {
            records_updated++;
            auto record = reader.nextRecord();
            vector<string> new_record;
            for (const auto &column: given_columns) {
                for (size_t i = 0; i < column->source_column_count; i++) {
                    new_record.push_back(record[column->start_index + i]);
                }
//                for (size_t i = 0; i < column.rows; i++) {
//                    for (size_t j = 0; j < column.columns; j++) {
//                        for (size_t k = 0; k < column.channels; k++) {
//                            size_t index = column.start_index + (i * column.columns * column.channels) + (j * column.channels) + k;
//                            new_record.push_back(record[index]);
//                        }
//                    }
//                }
            }
            for (const auto &column: expected_columns) {
                for (size_t i = 0; i < column->source_column_count; i++) {
                    new_record.push_back(record[column->start_index + i]);
                }
//                for (size_t i = 0; i < column.rows; i++) {
//                    for (size_t j = 0; j < column.columns; j++) {
//                        for (size_t k = 0; k < column.channels; k++) {
//                            size_t index = column.start_index + (i * column.columns * column.channels) + (j * column.channels) + k;
//                            new_record.push_back(record[index]);
//                        }
//                    }
//                }
            }
            writer.writeRecord(new_record);
        }
        writer.close();
        reader.close();
        cout << "Prepared " << records_updated << " records." << endl;
        return records_updated > 0;
    }
}

#endif //HAPPYML_TRAINING_DATASET_HPP
