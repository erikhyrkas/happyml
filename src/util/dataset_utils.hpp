//
// Created by Erik HYrkas on 4/15/2023.
//

#ifndef HAPPYML_DATASET_UTILS_HPP
#define HAPPYML_DATASET_UTILS_HPP

#include "../util/file_writer.hpp"
#include "../training_data/training_dataset.hpp"

namespace happyml {

    bool compare_startIndex(const ColumnGroup &a, const ColumnGroup &b) {
        return a.startIndex < b.startIndex;
    }

    bool has_overlap(const std::vector<ColumnGroup> &sorted_groups) {
        for (size_t i = 1; i < sorted_groups.size(); ++i) {
            const ColumnGroup &prev = sorted_groups[i - 1];
            const ColumnGroup &curr = sorted_groups[i];

            size_t prev_endIndex = prev.startIndex + (prev.rows * prev.columns * prev.channels);

            if (prev_endIndex > curr.startIndex) {
                return true;
            }
        }
        return false;
    }

    bool sort_and_check_overlaps(std::vector<ColumnGroup> &columnGroups) {
        std::sort(columnGroups.begin(), columnGroups.end(), compare_startIndex);
        return has_overlap(columnGroups);
    }

    unordered_set<string> get_distinct_values(const string &path,
                                              char delimiter,
                                              size_t columnIndex,
                                              bool header_row,
                                              bool trim_strings) {
        DelimitedTextFileReader delimitedTextFileReader(path, delimiter, header_row);

        unordered_set<string> distinctValues;
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



    void create_binary_dataset_from_delimited_values(const string &repo_path,
                                                     const string &name,
                                                     const string &path,
                                                     char delimiter,
                                                     bool header_row,
                                                     bool trim_strings,
                                                     vector<ColumnGroup> &sortedColumnGroups,
                                                     shared_ptr<BytePairEncoderModel> defaultBytePairEncoder) {


        vector<vector<string>> dataset_metadata;

        // image and number have simple encoders, but text and label need to be handled differently.
        // For labels, we need to create a unique category encoder for each column because
        // each column will have its own set of labels.
        // For text, we need to use byte pair encoding.
        for (auto column_group: sortedColumnGroups) {
            vector<string> column_group_metadata = {"column_group",
                                                    asString(column_group.expected),
                                                    column_group.encoder_name,
                                                    column_group.dataType,
                                                    to_string(column_group.startIndex),
                                                    to_string(column_group.rows),
                                                    to_string(column_group.columns),
                                                    to_string(column_group.channels)};
            dataset_metadata.push_back(column_group_metadata);

            if ("image" == column_group.encoder_name && column_group.encoder == nullptr) {
                column_group.encoder = make_shared<TextToPixelEncoder>();
            } else if ("number" == column_group.encoder_name) {
                column_group.encoder = make_shared<TextToScalarEncoder>();
            } else if ("label" == column_group.encoder_name) {
                auto distinctValues = get_distinct_values(path, delimiter, column_group.startIndex, header_row,
                                                          trim_strings);
                vector<string> distinctValuesVector(distinctValues.begin(), distinctValues.end());
                column_group.encoder = make_shared<TextToUniqueCategoryEncoder>(distinctValuesVector);
            } else if ("text" == column_group.encoder_name) {
//                column_group.encoder = make_shared<TextToEmbeddedTokensEncoder>(defaultBytePairEncoder);
                //TODO: finish this
            } else {
                // error we don't have an encoder for this type.
                break;
            }
        }
        auto new_dataset_path = repo_path + "/" + name;
        save_config(new_dataset_path, dataset_metadata);
        auto dataset = make_shared<InMemoryTrainingDataSet>();

        DelimitedTextFileReader delimitedTextFileReader(path, delimiter, header_row);


        // iterate over delimitedTextFileReader and create tensors for each column group.
        while (delimitedTextFileReader.hasNext()) {
            auto record = delimitedTextFileReader.nextRecord();
            for (auto column_group: sortedColumnGroups) {
                //TODO: finish this
            }
//            auto middleIter(record.begin());
//            std::advance(middleIter, expected_first ? expected_columns : given_columns);
//            vector<string> firstHalf(record.begin(), middleIter);
//            auto firstTensor = expected_first ?
//                               expected_encoder->encode(firstHalf, expected_shape[0], expected_shape[1],
//                                                        expected_shape[2], trim_strings) :
//                               given_encoder->encode(firstHalf, given_shape[0], given_shape[1], given_shape[2],
//                                                     trim_strings);
//            vector<string> secondHalf(middleIter, record.end());
//            auto secondTensor = expected_first ?
//                                given_encoder->encode(secondHalf, given_shape[0], given_shape[1], given_shape[2],
//                                                      trim_strings) :
//                                expected_encoder->encode(secondHalf, expected_shape[0], expected_shape[1],
//                                                         expected_shape[2], trim_strings);
//            if (expected_first) {
//                // given, expected
//                dataset->addTrainingData(secondTensor, firstTensor);
//            } else {
//                dataset->addTrainingData(firstTensor, secondTensor);
//            }
        }

//        return dataset;

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
