//
// Created by Erik Hyrkas on 5/5/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_PRINT_STATEMENT_HPP
#define HAPPYML_PRINT_STATEMENT_HPP

#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include "../execution_context.hpp"
#include "../../training_data/data_decoder.hpp"
#include "../../util/encoder_decoder_builder.hpp"

namespace happyml {

    class PrintStatement : public ExecutableStatement {
    public:
        explicit PrintStatement(string dataset_name, bool raw, int limit = -1) : dataset_name_(std::move(dataset_name)), raw_(raw), limit_(limit) {
        }

        static void print_display_rows(size_t max_display_rows, vector<vector<string>> &display_values) {
            for (size_t display_row = 0; display_row < max_display_rows; display_row++) {
                string delim;
                for (auto &display_value: display_values) {
                    if (!delim.empty()) {
                        cout << delim;
                    }
                    cout << display_value[display_row];
                    cout << " ";
                    delim = "|";
                }
                cout << endl;
            }
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            string result_path = context->get_dataset_path(dataset_name_) + "/dataset.bin";
            BinaryDatasetReader reader(result_path);
            auto row_count = reader.rowCount();
            auto max_result_rows = (limit_ == -1) ? reader.rowCount() : min(row_count, (size_t) limit_);
            cout << "Printing " << max_result_rows << " rows from dataset " << dataset_name_ << endl;

            vector<string> given_column_names = reader.get_given_names();
            vector<string> expected_column_names = reader.get_expected_names();
            //print given and expected column names
            cout << "Given: ";
            string delim;
            for (auto &given_column_name: given_column_names) {
                if (!delim.empty()) {
                    cout << delim;
                }
                cout << given_column_name;
                delim = "|";
            }
            cout << endl;
            cout << "Expected: ";
            delim = "";
            for (auto &expected_column_name: expected_column_names) {
                if (!delim.empty()) {
                    cout << delim;
                }
                cout << expected_column_name;
                delim = "|";
            }
            cout << endl;

            if (row_count == 0) {
                cout << "Dataset is empty." << endl;
                return make_shared<ExecutionResult>(false);
            }
            // auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
            vector<shared_ptr<RawDecoder>> given_decoders = build_given_decoders(raw_, reader);
            vector<shared_ptr<RawDecoder >> expected_decoders = build_expected_decoders(raw_, reader);

            for (int i = 0; i < max_result_rows; i++) {
                auto row = reader.readRow(i);
                auto given_tensors = row.first;
                auto expected_tensors = row.second;

                cout << "Row: " << (i + 1) << " Given: " << endl;
                size_t given_max_display_rows = 0;
                vector<vector<string >> given_display_values = calculate_display_values(given_max_display_rows,
                                                                                        given_tensors,
                                                                                        given_decoders);
                print_display_rows(given_max_display_rows, given_display_values);
                cout << "Row: " << (i + 1) << " Expected: " << endl;
                size_t expected_max_display_rows = 0;
                vector<vector<string >> expected_display_values = calculate_display_values(expected_max_display_rows,
                                                                                           expected_tensors,
                                                                                           expected_decoders);
                print_display_rows(expected_max_display_rows, expected_display_values);
            }

            return make_shared<ExecutionResult>(false);
        }


        static vector<vector<string>> calculate_display_values(size_t &max_display_rows,
                                                               vector<shared_ptr<BaseTensor>> &tensors_to_display,
                                                               vector<shared_ptr<RawDecoder>> &decoders) {
            vector<vector<string >> display_values;
            auto outer_offset = 0;
            for (auto &next_tensor: tensors_to_display) {
                size_t next_row_count;
                auto decoder = decoders[outer_offset];
                vector<string> next_values;
                if (decoder->isText()) {
                    next_row_count = 1;
                    string best = decoder->decodeBest(next_tensor);
                    next_values.push_back(best);
                } else {
                    next_row_count = next_tensor->rowCount();
                    auto corrected_tensor = decoder->decode(next_tensor);
                    for (size_t display_row = 0; display_row < next_row_count; display_row++) {
                        stringstream next_ss;
                        corrected_tensor->prettyPrintRow(next_ss, display_row);
                        next_values.push_back(next_ss.str());
                    }
                }
                display_values.push_back(next_values);
                max_display_rows = max(max_display_rows, next_row_count);
                outer_offset++;
            }
            return display_values;
        }


    private:
        string dataset_name_;
        int limit_;
        bool raw_;
    };
}
#endif //HAPPYML_PRINT_STATEMENT_HPP
