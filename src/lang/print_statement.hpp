//
// Created by Erik Hyrkas on 5/5/2023.
//

#ifndef HAPPYML_PRINT_STATEMENT_HPP
#define HAPPYML_PRINT_STATEMENT_HPP

#include "execution_context.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include "../util/happyml_paths.hpp"
#include "../training_data/data_decoder.hpp"

namespace happyml {

    class PrintStatement : public happyml::ExecutableStatement {
    public:
        explicit PrintStatement(string dataset_name, bool raw, int limit = -1) : dataset_name_(std::move(dataset_name)), raw_(raw), limit_(limit) {
        }

        static void print_display_rows(size_t max_display_rows, vector<vector<string >> &display_values) {
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

        shared_ptr<happyml::ExecutionResult> execute(const shared_ptr<happyml::ExecutionContext> &context) override {
            string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
            string result_path = base_path + dataset_name_ + "/dataset.bin";
            happyml::BinaryDatasetReader reader(result_path);
            auto row_count = reader.rowCount();
            auto max_result_rows = (limit_ == -1) ? reader.rowCount() : min(row_count, (size_t) limit_);
            cout << "Printing " << max_result_rows << " rows from dataset " << dataset_name_ << endl;

            if (row_count == 0) {
                cout << "Dataset is empty." << endl;
                return make_shared<happyml::ExecutionResult>(false);
            }
            // auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
            vector<shared_ptr<happyml::RawDecoder >> given_decoders;
            size_t given_column_count = reader.get_given_column_count();
            for (size_t i = 0; i < given_column_count; i++) {
                const shared_ptr<happyml::BinaryColumnMetadata> &metadata = reader.get_given_metadata(i);
                shared_ptr<happyml::RawDecoder> decoder = build_decoder(metadata);
                given_decoders.push_back(decoder);
            }
            vector<shared_ptr<happyml::RawDecoder >> expected_decoders;
            size_t expected_column_count = reader.get_expected_column_count();
            for (size_t i = 0; i < expected_column_count; i++) {
                const shared_ptr<happyml::BinaryColumnMetadata> &metadata = reader.get_expected_metadata(i);
                shared_ptr<happyml::RawDecoder> decoder = build_decoder(metadata);
                expected_decoders.push_back(decoder);
            }

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

            return make_shared<happyml::ExecutionResult>(false);
        }

        static vector<vector<string>> calculate_display_values(size_t &max_display_rows,
                                                               vector<shared_ptr<happyml::BaseTensor>> &tensors_to_display,
                                                               vector<shared_ptr<happyml::RawDecoder>> &decoders) {
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

        [[nodiscard]] shared_ptr<happyml::RawDecoder> build_decoder(const shared_ptr<happyml::BinaryColumnMetadata> &metadata) const {
            shared_ptr<happyml::RawDecoder> decoder;
            if (raw_) {
                decoder = make_shared<happyml::RawDecoder>();
            } else {
                auto purpose = metadata->purpose;
                // purpose: 'I' (image), 'T' (text), 'N' (number), 'L' (label)
                if ('L' == purpose) {
                    auto ordered_labels = metadata->ordered_labels;
                    decoder = make_shared<happyml::BestTextCategoryDecoder>(ordered_labels);
                } else if ('N' == purpose) {
                    decoder = make_shared<happyml::RawDecoder>(metadata->is_normalized,
                                                               metadata->is_standardized,
                                                               metadata->min_value,
                                                               metadata->max_value,
                                                               metadata->mean,
                                                               metadata->standard_deviation);
                } else {
                    decoder = make_shared<happyml::RawDecoder>();
                }
            }
            return decoder;
        }

    private:
        string dataset_name_;
        int limit_;
        bool raw_;
    };
}
#endif //HAPPYML_PRINT_STATEMENT_HPP
