//
// Created by Erik Hyrkas on 5/5/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CREATE_DATASET_STATEMENT_HPP
#define HAPPYML_CREATE_DATASET_STATEMENT_HPP

#include "../execution_context.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include "../../util/dataset_utils.hpp"
#include "../../util/text_file_sorter.hpp"
#include "../../util/happyml_paths.hpp"

namespace happyml {

    class CreateDatasetStatement : public happyml::ExecutableStatement {
    public:
        CreateDatasetStatement(string name,
                               string location,
                               bool has_header,
                               vector<shared_ptr<happyml::ColumnGroup>> column_groups) :

                name_(std::move(name)),
                location_(std::move(location)),
                has_header_(has_header),
                column_groups_(std::move(column_groups)) {
        }

        shared_ptr<happyml::ExecutionResult> execute(const shared_ptr<happyml::ExecutionContext> &context) override {
            string warning;

            if (location_.find("file://") != 0) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset only supports file:// location type at the moment.");
            }

            if (column_groups_.empty()) {
                // based on file extension, I could guess the column groups.
                // a txt file would have a single text column group
                // a csv or tsv is trickier. we could assume there the first column is the expected value, and the rest are given.
                // but that would be a guess. We'd also have to guess the data types. Expected might be a label, given might be a number.
                // for the moment, we'll make the caller be explicit.
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset must have at least one given column.");
            }

            bool has_text = false;
            for (const auto &columnGroup: column_groups_) {
                if (columnGroup->data_type != "label" &&
                    columnGroup->data_type != "number" &&
                    columnGroup->data_type != "text" &&
                    columnGroup->data_type != "image") {
                    if (columnGroup->use == "expected") {
                        return make_shared<happyml::ExecutionResult>(false, false,
                                                                     "create dataset's expected type must be one of: scalar, category, pixel, or text.");
                    } else {
                        return make_shared<happyml::ExecutionResult>(false, false,
                                                                     "create dataset's given type must be one of: scalar, category, pixel, or text.");
                    }
                } else if (columnGroup->data_type == "text" && !has_text) {
                    has_text = true;
                }
                if (columnGroup->use != "expected" && columnGroup->use != "given") {
                    return make_shared<happyml::ExecutionResult>(false, false,
                                                                 "create dataset's use must be one of: expected or given.");
                }
            }
            if (happyml::sort_and_check_overlaps(column_groups_)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset's utilize columns that overlap.");
            }

            if (has_text) {
                shared_ptr<happyml::BytePairEncoderModel> defaultBytePairEncoder = context->getBpeEncoder();
                if (defaultBytePairEncoder == nullptr) {
                    defaultBytePairEncoder = happyml::load_default_byte_pair_encoder(DEFAULT_HAPPYML_REPO_PATH);
                    context->setBpeEncoder(defaultBytePairEncoder);
                }
            }

            // TODO: remove any non-printable characters... maybe even uncommon ascii characters as well.
            // clean_dataset(location_, new_location, skip_header_, column_groups_, warning);


            // if file extension is txt, we need to convert it to csv. if the file extension is tsv, we need to convert it to csv.
            // if the file extension is csv, we can just use it as is.
            // if the file extension is anything else, we need to throw an error.
            size_t last_period_offset = location_.find_last_of('.');
            string file_extension = location_.substr(last_period_offset + 1);
            auto index_of_start_of_file_name = location_.find("://") + 3;
            string base_file_path = location_.substr(index_of_start_of_file_name, last_period_offset - index_of_start_of_file_name);
            string current_location = base_file_path + ".csv";
            bool has_header = has_header_;
            if (file_extension == "txt") {
                has_header = false;
                if (!happyml::convert_txt_to_csv(location_, current_location, 4000)) {
                    return make_shared<happyml::ExecutionResult>(false, false,
                                                                 "Could not open source or destination file to convert text to csv.");
                }
            } else if (file_extension == "tsv") {
                if (!happyml::convert_tsv_to_csv(location_, current_location)) {
                    return make_shared<happyml::ExecutionResult>(false, false,
                                                                 "Could not open source or destination file to convert tsv to csv.");
                }
            } else if (file_extension != "csv") {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset only supports .csv, .txt, and .tsv file types at the moment.");
            }
            if (!filesystem::exists(current_location)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset could not find the file: " + current_location);
            }
            vector<shared_ptr<happyml::ColumnGroup>>
                    given_column_groups;
            vector<shared_ptr<happyml::ColumnGroup>>
                    expected_column_groups;
            for (const auto &columnGroup: column_groups_) {
                if (columnGroup->use == "expected") {
                    expected_column_groups.push_back(columnGroup);
                } else {
                    given_column_groups.push_back(columnGroup);
                }
            }
            if (given_column_groups.empty()) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "create dataset must have at least one given column.");
            }
            vector<shared_ptr<happyml::ColumnGroup>>
                    updatedColumnGroups;
            size_t current_index = 0;
            for (const auto &columnGroup: given_column_groups) {
                auto updatedColumnGroup = make_shared<happyml::ColumnGroup>(columnGroup);
                updatedColumnGroup->start_index = current_index;
                current_index += updatedColumnGroup->source_column_count;
                updatedColumnGroups.push_back(updatedColumnGroup);
            }
            for (const auto &columnGroup: expected_column_groups) {
                auto updatedColumnGroup = make_shared<happyml::ColumnGroup>(columnGroup);
                updatedColumnGroup->start_index = current_index;
                current_index += updatedColumnGroup->source_column_count;
                updatedColumnGroups.push_back(updatedColumnGroup);
            }

            auto organized_location = base_file_path + ".given-expected.csv";
            if (!update_column_positions(current_location, organized_location, given_column_groups, expected_column_groups, has_header)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "Empty dataset");
            }

            auto sorted_location = base_file_path + ".sorted.csv";
            if (!happyml::FileSorter::sort(organized_location, sorted_location, false)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "Could not sort the given-expected file.");
            }
            if (!filesystem::remove(organized_location)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "Could not remove the given-expected file.");
            }

            auto raw_location = create_binary_dataset_from_delimited_values(DEFAULT_HAPPYML_DATASETS_PATH,
                                                                            name_,
                                                                            sorted_location,
                                                                            ',',
                                                                            false,
                                                                            updatedColumnGroups,
                                                                            column_groups_,
                                                                            context->getBpeEncoder());
            if (!filesystem::remove(sorted_location)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "Could not remove the sorted-deduped file.");
            }

            happyml::normalize_and_standardize_dataset(raw_location,
                                                       DEFAULT_HAPPYML_DATASETS_PATH,
                                                       name_);
            if (!filesystem::remove(raw_location)) {
                return make_shared<happyml::ExecutionResult>(false, false,
                                                             "Could not remove the clean dataset file.");
            }
            return make_shared<happyml::ExecutionResult>(false, true, "Created.");
        }

    private:
        string name_;
        string location_;
        vector<shared_ptr<happyml::ColumnGroup>> column_groups_;
        bool has_header_;
    };
}

#endif //HAPPYML_CREATE_DATASET_STATEMENT_HPP
