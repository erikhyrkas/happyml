//
// Created by Erik Hyrkas on 4/15/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_DATASET_UTILS_HPP
#define HAPPYML_DATASET_UTILS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <utility>
#include <unordered_set>
#include "../types/tensor_views/standardize_tensor_view.hpp"
#include "../types/tensor_views/normalize_tensor_view.hpp"
#include "../training_data/training_dataset.hpp"
#include "../types/tensor_views/unstandardize_tensor_view.hpp"
#include "../types/tensor_views/denormalize_tensor_view.hpp"

namespace happyml {

    bool compare_startIndex(const shared_ptr<ColumnGroup> &a, const shared_ptr<ColumnGroup> &b) {
        return a->start_index_ < b->start_index_;
    }

    bool has_overlap(const std::vector<shared_ptr<ColumnGroup>> &sorted_groups) {
        for (size_t i = 1; i < sorted_groups.size(); ++i) {
            const shared_ptr<ColumnGroup> &prev = sorted_groups[i - 1];
            const shared_ptr<ColumnGroup> &curr = sorted_groups[i];

            size_t prev_endIndex = prev->start_index_ + prev->source_column_count_ - 1;

            if (prev_endIndex > curr->start_index_) {
                return true;
            }
        }
        return false;
    }

    bool sort_and_check_overlaps(std::vector<shared_ptr<ColumnGroup>> &columnGroups) {
        std::sort(columnGroups.begin(), columnGroups.end(), compare_startIndex);
        return has_overlap(columnGroups);
    }

    unordered_set <string> get_distinct_values(const string &path,
                                               char delimiter,
                                               size_t columnIndex,
                                               bool header_row,
                                               bool trim_strings) {
        DelimitedTextFileReader delimitedTextFileReader(path, delimiter, header_row);

        unordered_set <string> distinctValues;
        while (delimitedTextFileReader.hasNext()) {
            auto record = delimitedTextFileReader.nextRecord();
            if (columnIndex < record.size()) {
                auto val = record[columnIndex];
                if (trim_strings) {
                    val = strip(val);
                }
                distinctValues.insert(val);
            }
        }
        return distinctValues;
    }

    shared_ptr<BytePairEncoderModel> load_default_byte_pair_encoder(const string &repo_path) {
        shared_ptr<BytePairEncoderModel> bytePairEncoderModel = make_shared<BytePairEncoderModel>();
        if (bytePairEncoderModel->load(repo_path, "default_token_encoder")) {
            return bytePairEncoderModel;
        }
        return nullptr;
    }

    bool convert_tsv_to_csv(const string &tsv_file_path, const string &csv_file_path) {
        DelimitedTextFileReader delimitedTextFileReader(tsv_file_path, '\t', false);
        DelimitedTextFileWriter delimitedTextFileWriter(csv_file_path, ',');
        if (!delimitedTextFileReader.is_open() || !delimitedTextFileWriter.is_open()) {
            return false;
        }
        while (delimitedTextFileReader.hasNext()) {
            delimitedTextFileWriter.writeRecord(delimitedTextFileReader.nextRecord());
        }
        delimitedTextFileReader.close();
        delimitedTextFileWriter.close();
        return true;
    }

    bool convert_txt_to_csv(const std::string &original_text_file_name, const std::string &new_csv_file_name, int character_limit) {
        std::ifstream input_file(original_text_file_name);
        DelimitedTextFileWriter delimitedTextFileWriter(new_csv_file_name, ',');

        if (!input_file.is_open() || !delimitedTextFileWriter.is_open()) {
            return false;
        }

        std::string token;
        std::string cell;

        int near_limit = character_limit * 4 / 5;
        bool write_cell;
        while (std::getline(input_file, token, ' ')) {
            if (cell.length() + token.length() + 1 <= character_limit) {
                cell += " " + token;
                auto token_len = token.length();
                write_cell = (cell.length() > near_limit && token_len > 1 && (token[token_len - 1] == '.' || token[token_len - 2] == '.'));
            } else {
                write_cell = true;
            }
            if (write_cell) {
                delimitedTextFileWriter.writeRecord({cell});
                cell = token;
            }
        }

        if (!cell.empty()) {
            delimitedTextFileWriter.writeRecord({cell});
        }

        input_file.close();
        delimitedTextFileWriter.close();
        return true;
    }


    shared_ptr<BaseTensor> &standardize_and_normalize(shared_ptr<BaseTensor> &standardized_normalized_given_tensor,
                                                      const shared_ptr<BinaryColumnMetadata> &current_metadata) {
        if (current_metadata->is_standardized) {
            standardized_normalized_given_tensor = make_shared<StandardizeTensorView>(standardized_normalized_given_tensor,
                                                                                      current_metadata->mean,
                                                                                      current_metadata->standard_deviation);
        }
        if (current_metadata->is_normalized) {
            standardized_normalized_given_tensor = make_shared<NormalizeTensorView>(standardized_normalized_given_tensor,
                                                                                    current_metadata->min_value,
                                                                                    current_metadata->max_value);
        }
        return
                standardized_normalized_given_tensor;
    }

    shared_ptr<BaseTensor> &unstandardize_and_denormalize(shared_ptr<BaseTensor> &unstandardized_denormalized_given_tensor, const shared_ptr<BinaryColumnMetadata> &current_metadata) {
        if (current_metadata->is_normalized) {
            unstandardized_denormalized_given_tensor = make_shared<DenormalizeTensorView>(unstandardized_denormalized_given_tensor,
                                                                                          current_metadata->min_value,
                                                                                          current_metadata->max_value);
        }
        if (current_metadata->is_standardized) {
            unstandardized_denormalized_given_tensor = make_shared<UnstandardizeTensorView>(unstandardized_denormalized_given_tensor,
                                                                                            current_metadata->mean,
                                                                                            current_metadata->standard_deviation);
        }
        return unstandardized_denormalized_given_tensor;
    }

    shared_ptr<BinaryColumnMetadata> initialize_column_metadata(const vector<size_t> &dims,
                                                                char purpose,
                                                                vector<string> ordered_labels,
                                                                string name) {
        auto next_metadata = make_shared<BinaryColumnMetadata>();
        next_metadata->purpose = purpose;
        next_metadata->rows = dims[0];
        next_metadata->columns = dims[1];
        next_metadata->channels = dims[2];
        next_metadata->is_standardized = false;
        next_metadata->mean = 0.0;
        next_metadata->standard_deviation = 0.0;
        next_metadata->is_normalized = false;
        next_metadata->min_value = 0.0;
        next_metadata->max_value = 0.0;
        next_metadata->ordered_labels = std::move(ordered_labels);
        next_metadata->name = std::move(name);
        return next_metadata;
    }

    void normalize_and_standardize_dataset(const string &raw_binary_file,
                                           const string &dataset_path) {

        BinaryDatasetReader binaryDatasetReader(raw_binary_file);

        bool given_has_numbers = false;
        vector<shared_ptr<BinaryColumnMetadata>> given_metadata;
        vector<shared_ptr<StandardizationAndNormalizationValues>> given_standardization_values;
        auto given_count = binaryDatasetReader.get_given_column_count();
        for (int i = 0; i < given_count; ++i) {
            shared_ptr<BinaryColumnMetadata> next_metadata = initialize_column_metadata(binaryDatasetReader.getGivenTensorDims(i),
                                                                                        binaryDatasetReader.getGivenTensorPurpose(i),
                                                                                        binaryDatasetReader.getGivenTensorOrderedLabels(i),
                                                                                        binaryDatasetReader.get_given_name(i));
            given_metadata.emplace_back(next_metadata);
            given_standardization_values.emplace_back(make_shared<StandardizationAndNormalizationValues>());
            if (given_metadata[i]->purpose == 'N') {
                given_has_numbers = true;
            }
        }
        bool expected_has_numbers = false;
        vector<shared_ptr<BinaryColumnMetadata>> expected_metadata;
        vector<shared_ptr<StandardizationAndNormalizationValues>> expected_standardization_values;
        auto expected_count = binaryDatasetReader.get_expected_column_count();
        for (int i = 0; i < expected_count; ++i) {
            shared_ptr<BinaryColumnMetadata> next_metadata = initialize_column_metadata(binaryDatasetReader.getExpectedTensorDims(i),
                                                                                        binaryDatasetReader.getExpectedTensorPurpose(i),
                                                                                        binaryDatasetReader.getExpectedTensorOrderedLabels(i),
                                                                                        binaryDatasetReader.get_expected_name(i));
            expected_metadata.emplace_back(next_metadata);
            expected_standardization_values.emplace_back(make_shared<StandardizationAndNormalizationValues>());
            if (expected_metadata[i]->purpose == 'N') {
                expected_has_numbers = true;
            }
        }

        if (given_has_numbers || expected_has_numbers) {
            size_t row_count = binaryDatasetReader.rowCount();
            for (size_t index = 0; index < row_count; ++index) {
                auto row = binaryDatasetReader.readRow(index);
                auto given_tensors = row.first;
                auto expected_tensors = row.second;
                if (given_has_numbers) {
                    for (int i = 0; i < given_count; i++) {
                        if (given_metadata[i]->purpose == 'N') {
                            update_standardization_normalization_values_calculation(given_standardization_values[i], given_tensors[i]);
                        }
                    }
                }
                if (expected_has_numbers) {
                    for (int i = 0; i < expected_count; i++) {
                        if (expected_metadata[i]->purpose == 'N') {
                            update_standardization_normalization_values_calculation(expected_standardization_values[i], expected_tensors[i]);
                        }
                    }
                }
            }
            if (given_has_numbers) {
                for (int i = 0; i < given_count; i++) {
                    bool is_numeric_value = given_metadata[i]->purpose == 'N';
                    given_metadata[i]->is_standardized = is_numeric_value && given_standardization_values[i]->standard_deviation > 1.0;
                    given_metadata[i]->mean = (float) given_standardization_values[i]->mean_result;
                    given_metadata[i]->standard_deviation = (float) given_standardization_values[i]->standard_deviation;
                    given_metadata[i]->is_normalized = is_numeric_value;
                    given_metadata[i]->min_value = (float) given_standardization_values[i]->min_value;
                    given_metadata[i]->max_value = (float) given_standardization_values[i]->max_value;
                }
            }
            if (expected_has_numbers) {
                for (int i = 0; i < expected_count; i++) {
                    bool is_numeric_value = expected_metadata[i]->purpose == 'N';
                    expected_metadata[i]->is_standardized = is_numeric_value && expected_standardization_values[i]->standard_deviation > 1.0;
                    expected_metadata[i]->mean = (float) expected_standardization_values[i]->mean_result;
                    expected_metadata[i]->standard_deviation = (float) expected_standardization_values[i]->standard_deviation;
                    expected_metadata[i]->is_normalized = is_numeric_value;
                    expected_metadata[i]->min_value = (float) expected_standardization_values[i]->min_value;
                    expected_metadata[i]->max_value = (float) expected_standardization_values[i]->max_value;
                }
            }
        }

        auto new_dataset_path = dataset_path + "/dataset.bin";
        BinaryDatasetWriter binaryDatasetWriter(new_dataset_path, given_metadata, expected_metadata, 0);
        size_t row_count = binaryDatasetReader.rowCount();
        for (size_t index = 0; index < row_count; ++index) {
            auto row = binaryDatasetReader.readRow(index);
            auto given_tensors = row.first;
            vector<shared_ptr<BaseTensor>> standardized_normalized_given_tensors;
            for (int i = 0; i < given_count; i++) {
                shared_ptr<BaseTensor> standardized_normalized_given_tensor = standardize_and_normalize(given_tensors[i], given_metadata[i]);
                standardized_normalized_given_tensors.emplace_back(standardized_normalized_given_tensor);
            }
            auto expected_tensors = row.second;
            vector<shared_ptr<BaseTensor>> standardized_normalized_expected_tensors;
            for (int i = 0; i < expected_count; i++) {
                shared_ptr<BaseTensor> standardized_normalized_expected_tensor = standardize_and_normalize(expected_tensors[i], expected_metadata[i]);
                standardized_normalized_expected_tensors.emplace_back(standardized_normalized_expected_tensor);
            }
            binaryDatasetWriter.writeRow(standardized_normalized_given_tensors, standardized_normalized_expected_tensors);
        }
        binaryDatasetWriter.close();
        binaryDatasetReader.close();
    }


    vector<string> build_column_group_metadata(const shared_ptr<ColumnGroup> &column_group, const string &metadata_label) {
        vector<string> column_group_metadata = {metadata_label,
                                                to_string(column_group->id_),
                                                to_string(column_group->start_index_),
                                                to_string(column_group->source_column_count_),
                                                column_group->use_,
                                                column_group->data_type_,
                                                column_group->label_,
                                                to_string(column_group->rows_),
                                                to_string(column_group->columns_),
                                                to_string(column_group->channels_)};
        return column_group_metadata;
    }

    void save_column_metadata(vector<pair<shared_ptr<ColumnGroup>, shared_ptr<DataEncoder>>> &columnGroupEncoders,
                              vector<shared_ptr<ColumnGroup>> &originalColumnGroups,
                              const string &new_dataset_path) {
        vector<vector<string>> dataset_metadata;
        for (const auto &column_group_encoders: columnGroupEncoders) {
            auto column_group = column_group_encoders.first;
            vector<string> column_group_metadata = build_column_group_metadata(column_group, "column_group");
            dataset_metadata.push_back(column_group_metadata);
        }
        for (const auto &column_group: originalColumnGroups) {
            vector<string> column_group_metadata = build_column_group_metadata(column_group, "original_column_group");
            dataset_metadata.push_back(column_group_metadata);
        }
        save_config(new_dataset_path, "dataset.config", dataset_metadata);
    }

    void save_column_metadata(vector<shared_ptr<ColumnGroup>> &columnGroups,
                              vector<shared_ptr<ColumnGroup>> &originalColumnGroups,
                              const string &new_dataset_path) {
        vector<vector<string>> dataset_metadata;
        for (const auto &column_group: columnGroups) {
            vector<string> column_group_metadata = build_column_group_metadata(column_group, "column_group");
            dataset_metadata.push_back(column_group_metadata);
        }
        for (const auto &column_group: originalColumnGroups) {
            vector<string> column_group_metadata = build_column_group_metadata(column_group, "original_column_group");
            dataset_metadata.push_back(column_group_metadata);
        }
        save_config(new_dataset_path, "dataset.config", dataset_metadata);
    }

    pair<vector<shared_ptr<ColumnGroup>>, vector<shared_ptr<ColumnGroup>>> read_column_metadata(const string &dataset_path) {
        auto dataset_metadata = read_config(dataset_path, "dataset.config");
        vector<shared_ptr<ColumnGroup>> sortedColumnGroups;
        vector<shared_ptr<ColumnGroup>> originalColumnGroups;
        for (const auto &column_group_metadata: dataset_metadata) {
            auto column_group = make_shared<ColumnGroup>();
            column_group->id_ = stoi(column_group_metadata[1]);
            column_group->start_index_ = stoi(column_group_metadata[2]);
            column_group->source_column_count_ = stoi(column_group_metadata[3]);
            column_group->use_ = column_group_metadata[4];
            column_group->data_type_ = column_group_metadata[5];
            column_group->label_ = column_group_metadata[6];
            column_group->rows_ = stoi(column_group_metadata[7]);
            column_group->columns_ = stoi(column_group_metadata[8]);
            column_group->channels_ = stoi(column_group_metadata[9]);
            if ("column_group" == column_group_metadata[0]) {
                sortedColumnGroups.emplace_back(column_group);
            } else if ("original_column_group" == column_group_metadata[0]) {
                originalColumnGroups.emplace_back(column_group);
            }
        }
        return make_pair(sortedColumnGroups, originalColumnGroups);

    }

    string create_binary_dataset_from_delimited_values(const string &new_dataset_path,
                                                       const string &delimited_file_path,
                                                       char delimiter,
                                                       bool header_row,
                                                       vector<shared_ptr<ColumnGroup>> &sortedColumnGroups,
                                                       vector<shared_ptr<ColumnGroup>> &originalColumnGroups,
                                                       const shared_ptr<BytePairEncoderModel> &defaultBytePairEncoder) {



        // image and number have simple encoders, but text and label need to be handled differently.
        // For labels, we need to create a unique category encoder for each column because
        // each column will have its own set of labels.
        // For text, we need to use byte pair encoding.
        vector<pair<shared_ptr<ColumnGroup>, shared_ptr<DataEncoder>>> columnGroupEncoders;
        for (auto &column_group: sortedColumnGroups) {
            if ("image" == column_group->data_type_) {
                columnGroupEncoders.emplace_back(make_shared<ColumnGroup>(*column_group), make_shared<TextToPixelEncoder>());
            } else if ("number" == column_group->data_type_) {
                columnGroupEncoders.emplace_back(make_shared<ColumnGroup>(*column_group), make_shared<TextToScalarEncoder>());
            } else if ("label" == column_group->data_type_) {
                auto distinctValues = get_distinct_values(delimited_file_path,
                                                          delimiter,
                                                          column_group->start_index_,
                                                          header_row,
                                                          true);
                vector<string> distinctValuesVector(distinctValues.begin(), distinctValues.end());
                columnGroupEncoders.emplace_back(make_shared<ColumnGroup>(column_group, distinctValuesVector), make_shared<TextToUniqueCategoryEncoder>(distinctValuesVector));
            } else if ("text" == column_group->data_type_) {
                //columnGroupEncoders.emplace_back(make_shared<ColumnGroup>(*column_group), make_shared<TextToEmbeddedTokensEncoder>(defaultBytePairEncoder));
                //TODO: finish this
                string error = "Unimplemented data type: " + column_group->data_type_;
                throw runtime_error(error.c_str());
            } else {
                string error = "Unknown data type: " + column_group->data_type_;
                throw runtime_error(error.c_str());
            }
            auto column_and_encoder = columnGroupEncoders.back();
            vector<size_t> shape = column_and_encoder.second->calculate_output_shape(column_and_encoder.first->rows_,
                                                                                     column_and_encoder.first->columns_,
                                                                                     column_and_encoder.first->channels_);
            // update the original shape, since it is wrong:
            column_and_encoder.first->rows_ = shape[0];
            column_and_encoder.first->columns_ = shape[1];
            column_and_encoder.first->channels_ = shape[2];
        }


        DelimitedTextFileReader delimitedTextFileReader(delimited_file_path, delimiter, header_row);

        vector<shared_ptr<BinaryColumnMetadata>> given_metadata;
        vector<shared_ptr<BinaryColumnMetadata>> expected_metadata;
        for (auto &column_group_plus_encoder: columnGroupEncoders) {
            auto next_column_metadata = make_shared<BinaryColumnMetadata>();
            // 'I' (image), 'T' (text), 'N' (number), 'L' (label)
            if (column_group_plus_encoder.first->data_type_ == "image") {
                next_column_metadata->purpose = 'I';
            } else if (column_group_plus_encoder.first->data_type_ == "text") {
                next_column_metadata->purpose = 'T';
            } else if (column_group_plus_encoder.first->data_type_ == "number") {
                next_column_metadata->purpose = 'N';
            } else if (column_group_plus_encoder.first->data_type_ == "label") {
                next_column_metadata->purpose = 'L';
            } else {
                throw std::runtime_error("Unknown data type: " + column_group_plus_encoder.first->data_type_);
            }
            next_column_metadata->source_column_count = column_group_plus_encoder.first->source_column_count_;
            next_column_metadata->rows = column_group_plus_encoder.first->rows_;
            next_column_metadata->columns = column_group_plus_encoder.first->columns_;
            next_column_metadata->channels = column_group_plus_encoder.first->channels_;
            next_column_metadata->is_normalized = false;
            next_column_metadata->min_value = 0.0;
            next_column_metadata->max_value = 0.0;
            next_column_metadata->is_standardized = false;
            next_column_metadata->mean = 0.0;
            next_column_metadata->standard_deviation = 0.0;
            next_column_metadata->ordered_labels = column_group_plus_encoder.first->ordered_distinct_labels_;
            next_column_metadata->name = column_group_plus_encoder.first->label_;
            if (column_group_plus_encoder.first->use_ == "given") {
                given_metadata.push_back(next_column_metadata);
            } else {
                expected_metadata.push_back(next_column_metadata);
            }
        }

        save_column_metadata(columnGroupEncoders, originalColumnGroups, new_dataset_path);

        auto dataset_file_path = new_dataset_path + "/raw.bin";
        BinaryDatasetWriter binaryDatasetWriter(dataset_file_path, given_metadata, expected_metadata);

        if (!binaryDatasetWriter.is_open()) {
            string error = "Unable to open dataset file for writing: " + dataset_file_path;
            throw runtime_error(error.c_str());
        }
        size_t row_count = 0;
        size_t written_count = 0;
        // iterate over delimitedTextFileReader and create tensors for each column group.
        while (delimitedTextFileReader.hasNext()) {
            row_count++;
            auto record = delimitedTextFileReader.nextRecord();
            vector<shared_ptr<BaseTensor>> result_row_givens;
            vector<shared_ptr<BaseTensor>> result_row_expecteds;

            for (const auto &column_group_and_encoder: columnGroupEncoders) {
                auto column_group_metadata = column_group_and_encoder.first;
                auto encoder = column_group_and_encoder.second;
                string data_type = column_group_metadata->data_type_;
                auto index_to_read_from = static_cast<vector<string>::difference_type>(column_group_metadata->start_index_);
                auto width = static_cast<vector<string>::difference_type>(index_to_read_from + (column_group_metadata->source_column_count_));
                size_t record_length = record.size();
                if (index_to_read_from >= record_length) {
                    throw std::runtime_error("Index to read from is greater than record length.  Index: " + to_string(index_to_read_from) + " Record Length: " + to_string(record_length));
                }
                if (record_length < width) {
                    throw std::runtime_error("Record length is less than expected.  Expected: " + to_string(width) + " Actual: " + to_string(record_length));
                }
                const _Vector_iterator<_Vector_val<_Simple_types<string>>> &start_iterator = record.begin() + index_to_read_from;
                const _Vector_iterator<_Vector_val<_Simple_types<string>>> &last_iterator = record.begin() + width;
                if (start_iterator > last_iterator) {
                    throw std::runtime_error("Start iterator is greater than last iterator.  Start: " + to_string(index_to_read_from) + " Last: " + to_string(width));
                }
                if (last_iterator > record.end()) {
                    throw std::runtime_error("Last iterator is greater than record end.  Last: " + to_string(width) + " Record End: " + to_string(record.end() - record.begin()));
                }
                vector<string> column_data(start_iterator, last_iterator);
                auto tensor_value = encoder->encode(column_data,
                                                    column_group_metadata->rows_,
                                                    column_group_metadata->columns_,
                                                    column_group_metadata->channels_,
                                                    true);
                // because columnGroupEncoders are in order, this should work
                if (column_group_metadata->use_ == "given") {
                    result_row_givens.push_back(tensor_value);
                } else {
                    result_row_expecteds.push_back(tensor_value);
                }
            }
            if (binaryDatasetWriter.writeRow(result_row_givens, result_row_expecteds)) {
                written_count++;
            }
        }
        binaryDatasetWriter.close();

        BinaryDatasetReader binaryDatasetReader(dataset_file_path);
        binaryDatasetReader.close();
        cout << "De-duped Text Rows: " << row_count << endl;
        cout << "De-duped Binary Rows written: " << written_count << endl;
        cout << "Binary Rows: " << binaryDatasetReader.rowCount() << endl;

        return dataset_file_path;
    }


    shared_ptr<InMemoryTrainingDataSet> loadDelimitedValuesDataset(const string &path, char delimiter,
                                                                   bool header_row, bool trim_strings,
                                                                   bool expected_first, size_t expected_columns,
                                                                   size_t given_columns,
                                                                   const vector<size_t> &expected_shape,
                                                                   const vector<size_t> &given_shape,
                                                                   const shared_ptr<DataEncoder> &expected_encoder,
                                                                   const shared_ptr<DataEncoder> &given_encoder) {

        auto dataset = make_shared<InMemoryTrainingDataSet>();

        DelimitedTextFileReader delimitedTextFileReader(path, delimiter, header_row);

        while (delimitedTextFileReader.hasNext()) {
            auto record = delimitedTextFileReader.nextRecord();
            auto middleIter(record.begin());
            std::advance(middleIter, expected_first ? expected_columns : given_columns);
            vector<string> firstHalf(record.begin(), middleIter);
            auto firstTensor = expected_first ?
                               expected_encoder->encode(firstHalf, expected_shape[0], expected_shape[1],
                                                        expected_shape[2], trim_strings) :
                               given_encoder->encode(firstHalf, given_shape[0], given_shape[1], given_shape[2],
                                                     trim_strings);
            vector<string> secondHalf(middleIter, record.end());
            auto secondTensor = expected_first ?
                                given_encoder->encode(secondHalf, given_shape[0], given_shape[1], given_shape[2],
                                                      trim_strings) :
                                expected_encoder->encode(secondHalf, expected_shape[0], expected_shape[1],
                                                         expected_shape[2], trim_strings);
            if (expected_first) {
                // given, expected
                dataset->addTrainingData(secondTensor, firstTensor);
            } else {
                dataset->addTrainingData(firstTensor, secondTensor);
            }
        }

        return dataset;
    }

}
#endif //HAPPYML_DATASET_UTILS_HPP
