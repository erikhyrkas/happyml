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

    class BinaryDataSet : public TrainingDataSet {
    public:
        explicit BinaryDataSet(const std::string &file_path,
                               const std::vector<shared_ptr<BinaryColumnMetadata>> &renormalize_given_metadata,
                               const std::vector<shared_ptr<BinaryColumnMetadata>> &renormalize_expected_metadata,
                               float split = 1.0f)
                : reader_(file_path, renormalize_given_metadata, renormalize_expected_metadata) {
            if (!reader_.is_open()) {
                throw std::runtime_error("Could not open file: " + file_path);
            }
            split_shuffler_ = make_shared<Shuffler>(reader_.rowCount());
            split_shuffler_->shuffle_predictably(); // shuffle once to make sure we have a fair distribution in case the original data was sorted.
            if (split > 0) {
                // left to right
                first_record_offset_ = 0;
                record_count_ = reader_.rowCount() * split;
            } else {
                // right to left
                first_record_offset_ = reader_.rowCount() * (1.0f + split);
                record_count_ = reader_.rowCount() - first_record_offset_;
            }
            current_offset_ = 0;
        }

        explicit BinaryDataSet(const std::string &file_path, float split = 1.0f) : reader_(file_path) {
            if (!reader_.is_open()) {
                throw std::runtime_error("Could not open file: " + file_path);
            }
            split_shuffler_ = make_shared<Shuffler>(reader_.rowCount());
            split_shuffler_->shuffle_predictably(); // shuffle once to make sure we have a fair distribution in case the original data was sorted.
            if (split > 0) {
                // left to right
                first_record_offset_ = 0;
                record_count_ = reader_.rowCount() * split;
            } else {
                // right to left
                first_record_offset_ = reader_.rowCount() * (1.0f + split);
                record_count_ = reader_.rowCount() - first_record_offset_;
            }
            current_offset_ = 0;
        }

        std::vector<shared_ptr<BinaryColumnMetadata>> getGivenMetadata() {
            return reader_.get_given_metadata();
        }

        std::vector<shared_ptr<BinaryColumnMetadata>> getExpectedMetadata() {
            return reader_.get_expected_metadata();
        }

        size_t recordCount() override {
            if (!reader_.is_open()) {
                return 0;
            }
            return record_count_;
        }

        void restart() override {
            current_offset_ = 0;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            if (current_offset_ >= recordCount()) {
                return nullptr;
            }
            auto shuffled_offset = shuffler_ != nullptr ? shuffler_->getShuffledIndex(current_offset_) : current_offset_;
            // we have to shuffle an extra time because we are splitting the dataset and the original data may be ordered
            // we want our split to have a fair distribution of data. The split shuffler is never reshuffled.
            auto split_shuffled_offset = split_shuffler_->getShuffledIndex(shuffled_offset);
            auto pair = reader_.readRow(shuffled_offset + first_record_offset_);
            current_offset_++;
            auto result = make_shared<TrainingPair>(pair.first, pair.second);
            return result;
        }

        vector<vector<size_t>> getGivenShapes() override {
            if (!reader_.is_open()) {
                throw std::runtime_error("Dataset file is not open");
            }
            vector<vector<size_t>> result;
            size_t given_count = reader_.get_given_column_count();
            for (size_t i = 0; i < given_count; i++) {
                result.push_back(reader_.getGivenTensorDims(i));
            }
            return result;
        }

        vector<vector<size_t>> getExpectedShapes() override {
            if (!reader_.is_open()) {
                throw std::runtime_error("Dataset file is not open");
            }
            vector<vector<size_t>> result;
            size_t expected_count = reader_.get_expected_column_count();
            for (size_t i = 0; i < expected_count; i++) {
                result.push_back(reader_.getExpectedTensorDims(i));
            }
            return result;
        }



    private:
        BinaryDatasetReader reader_;
        size_t current_offset_ = 0;
        size_t first_record_offset_ = 0;
        size_t record_count_ = 0;
        shared_ptr<Shuffler> split_shuffler_;
    };

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
        ColumnGroup() : id_(0), start_index_(0), source_column_count_(0), label_(), rows_(0), columns_(0), channels_(0) {}

        ColumnGroup(size_t id, size_t startIndex, size_t source_column_count, string use, string dataType, string label, size_t rows, size_t columns, size_t channels) :
                id_(id), start_index_(startIndex), source_column_count_(source_column_count), use_(std::move(use)), data_type_(std::move(dataType)),
                label_(std::move(label)), rows_(rows), columns_(columns), channels_(channels) {}

        explicit ColumnGroup(const shared_ptr<ColumnGroup> &from) {
            id_ = from->id_;
            start_index_ = from->start_index_;
            source_column_count_ = from->source_column_count_;
            use_ = from->use_;
            data_type_ = from->data_type_;
            rows_ = from->rows_;
            columns_ = from->columns_;
            channels_ = from->channels_;
            label_ = from->label_;
            ordered_distinct_labels_ = from->ordered_distinct_labels_;
        }

        ColumnGroup(const shared_ptr<ColumnGroup> &from, vector<string> ordered_distinct_labels) {
            id_ = from->id_;
            start_index_ = from->start_index_;
            source_column_count_ = from->source_column_count_;
            use_ = from->use_;
            data_type_ = from->data_type_;
            rows_ = from->rows_;
            columns_ = from->columns_;
            channels_ = from->channels_;
            label_ = from->label_;
            ordered_distinct_labels_ = std::move(ordered_distinct_labels);
        }

        size_t start_index_;
        size_t source_column_count_;
        string use_; // given or expected
        string data_type_; // image, label, number, text
        size_t rows_;
        size_t columns_;
        size_t channels_;
        size_t id_;
        string label_;
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
                for (size_t i = 0; i < column->source_column_count_; i++) {
                    new_record.push_back(record[column->start_index_ + i]);
                }
            }
            for (const auto &column: expected_columns) {
                for (size_t i = 0; i < column->source_column_count_; i++) {
                    new_record.push_back(record[column->start_index_ + i]);
                }
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
