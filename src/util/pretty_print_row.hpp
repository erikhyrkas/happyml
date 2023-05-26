//
// Created by Erik Hyrkas on 5/24/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_PRETTY_PRINT_ROW_HPP
#define HAPPYML_PRETTY_PRINT_ROW_HPP

#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include "../training_data/data_decoder.hpp"
#include "../util/encoder_decoder_builder.hpp"

namespace happyml {
    void pretty_print_header(ostream &stream, std::vector<std::string> &column_names, std::vector<std::size_t> &widths) {
        // print column names separated by |, padded with spaces to the width of the column
        string delim;
        for (size_t column_index = 0; column_index < column_names.size(); column_index++) {
            if (!delim.empty()) {
                stream << delim;
            }
            stream << std::setw((streamsize) widths[column_index]) << column_names[column_index];
            delim = "|";
        }
        stream << endl;
    }

    std::vector<std::size_t> calculate_pretty_print_column_widths(std::vector<std::string> &column_names, std::vector<std::vector<std::string >> &row_values) {
        if (column_names.size() != row_values.size()) {
            string message = "Column names has size " + std::to_string(column_names.size()) + " but row values has size " + std::to_string(row_values.size());
            throw runtime_error(message);
        }
        std::vector<size_t> widths;
        for (size_t column_index = 0; column_index < column_names.size(); column_index++) {
            size_t max_width = column_names[column_index].size();
            for (auto &row_value: row_values[column_index]) {
                max_width = std::max(max_width, row_value.size());
            }
            widths.push_back(max_width);
        }
        return widths;
    }

    void pretty_print_row(ostream &stream, std::vector<std::vector<std::string>> &row_values, std::vector<std::size_t> &widths) {
        // a result "row" can span multiple lines, but each element of row values may not be the same number of lines
        // print row are separated by |, padded with spaces to the width of the column
        size_t max_height = 0;
        for (auto &row_value: row_values) {
            max_height = std::max(max_height, row_value.size());
        }
        for (size_t current_height = 0; current_height < max_height; current_height++) {
            string delim;
            delim = "";
            size_t column_index = 0;
            for (auto &row_value: row_values) {
                size_t width = widths[column_index];
                column_index++;
                string value;
                if (current_height < row_value.size()) {
                    value = row_value[current_height];
                }
                if (!delim.empty()) {
                    stream << delim;
                }
                stream << std::setw((streamsize) width) << value;
                delim = "|";
            }

            stream << endl;
        }
    }

    std::vector<std::string> record_to_strings(shared_ptr<RawDecoder> &decoder, shared_ptr<BaseTensor> &record) {
        std::vector<std::string> result;
        if (decoder->isText()) {
            string best = decoder->decodeBest(record);
            result.push_back(best);
        } else {
            auto row_count = record->rowCount();
            auto corrected_tensor = decoder->decode(record);
            for (size_t display_row = 0; display_row < row_count; display_row++) {
                stringstream next_ss;
                corrected_tensor->prettyPrintRow(next_ss, display_row);
                result.push_back(next_ss.str());
            }
        }
        return result;
    }

    std::vector<std::vector<std::string>> record_group_to_strings(std::vector<shared_ptr<RawDecoder>> &decoders, std::vector<shared_ptr<BaseTensor>> &record_group) {
        std::vector<std::vector<std::string>> result;
        for (size_t record_index = 0; record_index < record_group.size(); record_index++) {
            auto record = record_group[record_index];
            auto decoder = decoders[record_index];
            auto next_values = record_to_strings(decoder, record);
            result.push_back(next_values);
        }
        return result;
    }

    std::vector<std::vector<std::string>> pretty_print_merge_records(std::vector<shared_ptr<RawDecoder>> &expected_decoders, std::vector<shared_ptr<BaseTensor>> &expected_record_group,
                                                                     std::vector<shared_ptr<RawDecoder>> &given_decoders, std::vector<shared_ptr<BaseTensor>> &given_record_group) {

        std::vector<std::vector<std::string>> result;
        auto expected_values = record_group_to_strings(expected_decoders, expected_record_group);
        auto given_values = record_group_to_strings(given_decoders, given_record_group);
        result.reserve(expected_values.size() + given_values.size());
        for (const auto &expected_value: expected_values) {
            result.push_back(expected_value);
        }
        for (const auto &given_value: given_values) {
            result.push_back(given_value);
        }
        return result;
    }

    std::vector<std::string> pretty_print_merge_headers(const std::vector<std::string> &expected, const std::vector<std::string> &given) {
        std::vector<std::string> result;
        result.reserve(given.size() + expected.size());
        for (const auto &expected_value: expected) {
            result.push_back(expected_value);
        }
        for (const auto &given_value: given) {
            result.push_back(given_value);
        }
        return result;
    }

    void pretty_print(ostream &stream, BinaryDatasetReader &reader, int limit = -1, bool raw = false) {
        auto row_count = reader.rowCount();
        auto max_result_rows = (limit == -1) ? reader.rowCount() : min(row_count, (size_t) limit);
        vector<string> given_column_names = reader.get_given_names();
        vector<string> expected_column_names = reader.get_expected_names();
        vector<shared_ptr<RawDecoder>> given_decoders = build_given_decoders(raw, reader);
        vector<shared_ptr<RawDecoder>> expected_decoders = build_expected_decoders(raw, reader);
        vector<string> merged_headers = pretty_print_merge_headers(expected_column_names, given_column_names);
        vector<size_t> widths;
        for (size_t row_index = 0; row_index < max_result_rows; row_index++) {
            auto row = reader.readRow(row_index);
            auto given_record_group = row.first;
            auto expected_record_group = row.second;

            auto merged_values = pretty_print_merge_records(expected_decoders, expected_record_group, given_decoders, given_record_group);
            if (widths.empty()) {
                widths = calculate_pretty_print_column_widths(merged_headers, merged_values);
                pretty_print_header(stream, merged_headers, widths);
            }
            pretty_print_row(stream, merged_values, widths);
        }

    }
}
#endif //HAPPYML_PRETTY_PRINT_ROW_HPP
