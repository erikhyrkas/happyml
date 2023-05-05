//
// Created by Erik Hyrkas on 3/25/2023.
//

#ifndef HAPPYML_STATEMENTS_HPP
#define HAPPYML_STATEMENTS_HPP


#include <string>
#include <iostream>
#include <utility>
#include <vector>
#include "../util/happyml_paths.hpp"
#include "execution_context.hpp"
#include "../training_data/training_dataset.hpp"
#include "../util/dataset_utils.hpp"
#include "../util/text_file_sorter.hpp"
#include "../training_data/data_decoder.hpp"

using namespace std;

namespace happyml {

    class ExitStatement : public ExecutableStatement {
    public:
        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            cout << "Exiting..." << endl;
            return make_shared<ExecutionResult>(true);
        }
    };

    class PrintStatement : public ExecutableStatement {
    public:
        explicit PrintStatement(string dataset_name, bool raw, int limit = -1) : dataset_name_(std::move(dataset_name)), raw_(raw), limit_(limit) {
        }

        void print_display_rows(size_t max_display_rows, vector <vector<string >> &display_values) const {
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

        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
            string result_path = base_path + dataset_name_ + "/dataset.bin";
            BinaryDatasetReader reader(result_path);
            auto row_count = reader.rowCount();
            auto max_result_rows = (limit_ == -1) ? reader.rowCount() : min(row_count, (size_t) limit_);
            cout << "Printing " << max_result_rows << " rows from dataset " << dataset_name_ << endl;

            if (row_count == 0) {
                cout << "Dataset is empty." << endl;
                return make_shared<ExecutionResult>(false);
            }
            // auto decoder = make_shared<BestTextCategoryDecoder>(categoryLabels);
            vector < shared_ptr < RawDecoder >> given_decoders;
            size_t given_column_count = reader.get_given_column_count();
            for (size_t i = 0; i < given_column_count; i++) {
                const shared_ptr <BinaryColumnMetadata> &metadata = reader.get_given_metadata(i);
                shared_ptr < RawDecoder > decoder = build_decoder(metadata);
                given_decoders.push_back(decoder);
            }
            vector < shared_ptr < RawDecoder >> expected_decoders;
            size_t expected_column_count = reader.get_expected_column_count();
            for (size_t i = 0; i < expected_column_count; i++) {
                const shared_ptr <BinaryColumnMetadata> &metadata = reader.get_expected_metadata(i);
                shared_ptr < RawDecoder > decoder = build_decoder(metadata);
                expected_decoders.push_back(decoder);
            }

            for (int i = 0; i < max_result_rows; i++) {
                auto row = reader.readRow(i);
                auto given_tensors = row.first;
                auto expected_tensors = row.second;

                cout << "Row: " << (i + 1) << " Given: " << endl;
                size_t given_max_display_rows = 0;
                vector < vector < string >> given_display_values = calculate_display_values(given_max_display_rows,
                                                                                            given_tensors,
                                                                                            given_decoders);
                print_display_rows(given_max_display_rows, given_display_values);
                cout << "Row: " << (i + 1) << " Expected: " << endl;
                size_t expected_max_display_rows = 0;
                vector < vector < string >> expected_display_values = calculate_display_values(expected_max_display_rows,
                                                                                               expected_tensors,
                                                                                               expected_decoders);
                print_display_rows(expected_max_display_rows, expected_display_values);
            }

            return make_shared<ExecutionResult>(false);
        }

        static vector <vector<string>> calculate_display_values(size_t &max_display_rows,
                                                                vector <shared_ptr<BaseTensor>> &tensors_to_display,
                                                                vector <shared_ptr<RawDecoder>> &decoders) {
            vector < vector < string >> display_values;
            auto outer_offset = 0;
            for (auto &next_tensor: tensors_to_display) {
                size_t next_row_count;
                auto decoder = decoders[outer_offset];
                vector < string > next_values;
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

        [[nodiscard]] shared_ptr <RawDecoder> build_decoder(const shared_ptr <BinaryColumnMetadata> &metadata) const {
            shared_ptr < RawDecoder > decoder;
            if (raw_) {
                decoder = make_shared<RawDecoder>();
            } else {
                auto purpose = metadata->purpose;
                // purpose: 'I' (image), 'T' (text), 'N' (number), 'L' (label)
                if ('L' == purpose) {
                    auto ordered_labels = metadata->ordered_labels;
                    decoder = make_shared<BestTextCategoryDecoder>(ordered_labels);
                } else if ('N' == purpose) {
                    decoder = make_shared<RawDecoder>(metadata->is_normalized,
                                                      metadata->is_standardized,
                                                      metadata->min_value,
                                                      metadata->max_value,
                                                      metadata->mean,
                                                      metadata->standard_deviation);
                } else {
                    decoder = make_shared<RawDecoder>();
                }
            }
            return decoder;
        }

    private:
        string dataset_name_;
        int limit_;
        bool raw_;
    };

    class HelpStatement : public ExecutableStatement {
    public:
        explicit HelpStatement(string help_menu_item = "default") : help_menu_item_(std::move(help_menu_item)) {

        }

        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {

            if (help_menu_item_ == "dataset" || help_menu_item_ == "datasets") {
                cout << "Available dataset commands: " << endl;
                cout << "  create dataset <name>" << endl
                     << "  [with header]" << endl
                     << "  [with given <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]+ " << endl
                     << "  [with expected <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]*" << endl
                     << "  using <file://path/>" << endl << endl;

                cout << "  print pretty <name> [limit <x>]" << endl << endl;
                cout << "  print raw <name> [limit <x>]" << endl << endl;

            } else if (help_menu_item_ == "task" || help_menu_item_ == "tasks") {
                cout << "Available task commands: " << endl;

                cout << "  coming soon..." << endl << endl;

            } else if (help_menu_item_ == "future") {
                cout << "Future commands: " << endl;
                cout << "  create task <task type> <task name>" << endl
                     << "  [with goal <speed|accuracy|memory>]" << endl
                     << "  using <dataset name>" << endl << endl;

                cout << "  execute task <task name>" << endl
                     << "  [with label <label>]" << endl
                     << "  using dataset <dataset>" << endl << endl;

                cout << "  list tasks [<starting with x>]" << endl << endl;

                cout << "  refine task <task name>" << endl
                     << "  [with label [label]]" << endl
                     << "  using dataset <dataset name>" << endl << endl;

                cout << "  list datasets [<starting with x>]" << endl << endl;

                cout << "  copy <task name> [<label>] to [<task name>] [<label>]" << endl << endl;

                cout << "  copy <dataset name> to [<dataset name>]" << endl << endl;

                cout << "  delete <task name> [<label>]" << endl << endl;

                cout << "  delete <dataset name>" << endl << endl;

                cout << "  execute task <task name>" << endl
                     << "  [with label <label>]" << endl
                     << "  using input <csv encoded row>" << endl << endl;

                cout << "  move <task name> [<label>] to [<task name>] [<label>]" << endl << endl;

                cout << "  move <dataset name> to [<dataset name>] [<label>]" << endl << endl;
            } else {
                cout << "Available commands: " << endl;
                cout << "  exit" << endl << endl;

                cout << "  help [dataset|task|future]" << endl << endl;
            }

            return make_shared<ExecutionResult>(false);
        }

    private:
        string help_menu_item_;
    };


    class CreateDatasetStatement : public ExecutableStatement {
    public:
        CreateDatasetStatement(string name,
                               string location,
                               bool has_header,
                               vector<shared_ptr < ColumnGroup>>

        column_groups) :

        name_ (std::move(name)),
        location_(std::move(location)),
        has_header_(has_header),
        column_groups_(std::move(column_groups)) {
        }


        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            string warning;

            if (location_.find("file://") != 0) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset only supports file:// location type at the moment.");
            }

            if (column_groups_.empty()) {
                // based on file extension, I could guess the column groups.
                // a txt file would have a single text column group
                // a csv or tsv is trickier. we could assume there the first column is the expected value, and the rest are given.
                // but that would be a guess. We'd also have to guess the data types. Expected might be a label, given might be a number.
                // for the moment, we'll make the caller be explicit.
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset must have at least one given column.");
            }

            bool has_text = false;
            for (const auto &columnGroup: column_groups_) {
                if (columnGroup->data_type != "label" &&
                    columnGroup->data_type != "number" &&
                    columnGroup->data_type != "text" &&
                    columnGroup->data_type != "image") {
                    if (columnGroup->use == "expected") {
                        return make_shared<ExecutionResult>(false, false,
                                                            "create dataset's expected type must be one of: scalar, category, pixel, or text.");
                    } else {
                        return make_shared<ExecutionResult>(false, false,
                                                            "create dataset's given type must be one of: scalar, category, pixel, or text.");
                    }
                } else if (columnGroup->data_type == "text" && !has_text) {
                    has_text = true;
                }
                if (columnGroup->use != "expected" && columnGroup->use != "given") {
                    return make_shared<ExecutionResult>(false, false,
                                                        "create dataset's use must be one of: expected or given.");
                }
            }
            if (sort_and_check_overlaps(column_groups_)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset's utilize columns that overlap.");
            }

            if (has_text) {
                shared_ptr < BytePairEncoderModel > defaultBytePairEncoder = context->getBpeEncoder();
                if (defaultBytePairEncoder == nullptr) {
                    defaultBytePairEncoder = load_default_byte_pair_encoder(DEFAULT_HAPPYML_REPO_PATH);
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
                if (!convert_txt_to_csv(location_, current_location, 4000)) {
                    return make_shared<ExecutionResult>(false, false,
                                                        "Could not open source or destination file to convert text to csv.");
                }
            } else if (file_extension == "tsv") {
                if (!convert_tsv_to_csv(location_, current_location)) {
                    return make_shared<ExecutionResult>(false, false,
                                                        "Could not open source or destination file to convert tsv to csv.");
                }
            } else if (file_extension != "csv") {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset only supports .csv, .txt, and .tsv file types at the moment.");
            }
            if (!filesystem::exists(current_location)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset could not find the file: " + current_location);
            }
            vector<shared_ptr < ColumnGroup>>
            given_column_groups;
            vector<shared_ptr < ColumnGroup>>
            expected_column_groups;
            for (const auto &columnGroup: column_groups_) {
                if (columnGroup->use == "expected") {
                    expected_column_groups.push_back(columnGroup);
                } else {
                    given_column_groups.push_back(columnGroup);
                }
            }
            if (given_column_groups.empty()) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset must have at least one given column.");
            }
            vector<shared_ptr < ColumnGroup>>
            updatedColumnGroups;
            size_t current_index = 0;
            for (const auto &columnGroup: given_column_groups) {
                auto updatedColumnGroup = make_shared<ColumnGroup>(columnGroup);
                updatedColumnGroup->start_index = current_index;
                current_index += updatedColumnGroup->source_column_count;
                updatedColumnGroups.push_back(updatedColumnGroup);
            }
            for (const auto &columnGroup: expected_column_groups) {
                auto updatedColumnGroup = make_shared<ColumnGroup>(columnGroup);
                updatedColumnGroup->start_index = current_index;
                current_index += updatedColumnGroup->source_column_count;
                updatedColumnGroups.push_back(updatedColumnGroup);
            }

            auto organized_location = base_file_path + ".given-expected.csv";
            if (!update_column_positions(current_location, organized_location, given_column_groups, expected_column_groups, has_header)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "Empty dataset");
            }

            auto sorted_location = base_file_path + ".sorted.csv";
            if (!FileSorter::sort(organized_location, sorted_location, false)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "Could not sort the given-expected file.");
            }
            if (!filesystem::remove(organized_location)) {
                return make_shared<ExecutionResult>(false, false,
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
                return make_shared<ExecutionResult>(false, false,
                                                    "Could not remove the sorted-deduped file.");
            }

            normalize_and_standardize_dataset(raw_location,
                                              DEFAULT_HAPPYML_DATASETS_PATH,
                                              name_);
            if (!filesystem::remove(raw_location)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "Could not remove the clean dataset file.");
            }
            return make_shared<ExecutionResult>(false, true, "Created.");
        }

    private:
        string name_;
        string location_;
        vector<shared_ptr < ColumnGroup>> column_groups_;
        bool has_header_;
    };

    class CodeBlock : public ExecutableStatement {
    public:
        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr < ExecutionResult > lastResult = make_shared<ExecutionResult>();
            for (const auto &child: children) {
                lastResult = child->execute(context);
                // We are discarding all results but the last one. This is fine for handling errors, but
                // I'm not sure if we should use them for anything. I don't need them now, so this is fine.
                if (!lastResult->isSuccessful()) {
                    break;
                }
            }
            return lastResult;
        }

        void addChild(const shared_ptr <ExecutableStatement> &child) {
            children.push_back(child);
        }

    private:
        vector<shared_ptr < ExecutableStatement>> children{};
    };
}
#endif //HAPPYML_STATEMENTS_HPP
