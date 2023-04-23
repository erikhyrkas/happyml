//
// Created by Erik Hyrkas on 3/25/2023.
//

#ifndef HAPPYML_STATEMENTS_HPP
#define HAPPYML_STATEMENTS_HPP

#define DEFAULT_HAPPYML_REPO_PATH "../happyml_repo/"

#include <string>
#include <iostream>
#include <utility>
#include <vector>
#include "execution_context.hpp"
#include "../training_data/training_dataset.hpp"
#include "../util/dataset_utils.hpp"

using namespace std;

namespace happyml {

    class ExitStatement : public ExecutableStatement {
    public:
        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            cout << "Exiting..." << endl;
            return make_shared<ExecutionResult>(true);
        }
    };

    class HelpStatement : public ExecutableStatement {
    public:
        explicit HelpStatement(string help_menu_item = "default") : help_menu_item_(std::move(help_menu_item)) {

        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {

            if (help_menu_item_ == "dataset" || help_menu_item_ == "datasets") {
                cout << "Available dataset commands: " << endl;
                cout << "  create dataset <name>" << endl
                     << "  [with expected <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]*"
                     << endl
                     << "  [with given <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]* "
                     << endl
                     << "  using <file://path/>" << endl << endl;

                cout << "  list datasets [<starting with x>]" << endl << endl;

            } else if (help_menu_item_ == "task" || help_menu_item_ == "tasks") {
                cout << "Available task commands: " << endl;
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

            } else if (help_menu_item_ == "future") {
                cout << "Future commands: " << endl;
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
                               bool skip_header,
                               vector<ColumnGroup> column_groups) :
                name_(std::move(name)),
                location_(std::move(location)),
                skip_header_(skip_header),
                column_groups_(std::move(column_groups)) {
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            string warning;

            // default to success if there are no children.
            if (location_.find("file://") != 0) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset only supports file:// location type at the moment.");
            }
            bool has_text = false;
            for (const auto &columnGroup: column_groups_) {
                if (columnGroup.dataType != "label" &&
                    columnGroup.dataType != "number" &&
                    columnGroup.dataType != "text" &&
                    columnGroup.dataType != "image") {
                    if (columnGroup.expected) {
                        return make_shared<ExecutionResult>(false, false,
                                                            "create dataset's expected type must be one of: scalar, category, pixel, or text.");
                    } else {
                        return make_shared<ExecutionResult>(false, false,
                                                            "create dataset's given type must be one of: scalar, category, pixel, or text.");
                    }
                } else if (columnGroup.dataType == "text" && !has_text) {
                    has_text = true;
                }
            }
            if (sort_and_check_overlaps(column_groups_)) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset's utilize columns that overlap.");
            }

            if (has_text) {
                shared_ptr<BytePairEncoderModel> defaultBytePairEncoder = context->getBpeEncoder();
                if (defaultBytePairEncoder == nullptr) {
                    defaultBytePairEncoder = load_default_byte_pair_encoder(DEFAULT_HAPPYML_REPO_PATH);
                    context->setBpeEncoder(defaultBytePairEncoder);
                }
            }

            // steps:
            // 1. Map old column order to new column order with givens values first, then expected values second, we'll need this for the following steps

            vector<ColumnGroup> given_column_groups;
            vector<ColumnGroup> expected_column_groups;
            for( const auto &columnGroup : column_groups_ ) {
                if( columnGroup.expected ) {
                    expected_column_groups.push_back( columnGroup );
                } else {
                    given_column_groups.push_back( columnGroup );
                }
            }

            // 2. Copy the original text file to a new "given-expected" file, arranging given before expected values
            //    b. Copy the original file to a new file, arranging columns as expected

            update_column_positions(location_, location_ + ".given-expected", given_column_groups, expected_column_groups, skip_header_ );


            // 3. Use FileSorter.sort() to sort and dedupe the "given-expected" file as a new "sorted-deduped" file
            //    NOTE: remove "given-expected" file


            // 4. Create new BinaryDataset from "sorted-deduped" file, deduping any givens that are the same, call new binary file "clean dataset"
            //    NOTE: remove "sorted-deduped" file
            // 5. If needed, standarize and normalize the "clean dataset" file, call new binary file "standardized-normalized dataset"
            //    NOTE: remove "clean dataset" file
            // 6. Save a properties file that tracks the original given and expected metadata (data types, shapes, and starting column positions) and the new binary order
            //    NOTE: This is used later when the user creates a task, and the task needs to understand how the user might send requests



            //TODO: finish this
//            create_binary_dataset_from_delimited_values(DEFAULT_HAPPYML_REPO_PATH,
//                                                        name,
//                                                        location);
            //  create dataset <name>
            //  [with expected <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]*
            //  [with given <label|number|text|image> [(<rows>, <columns>, <channels>)] at <column> ]*
            //  using <file://path/>

//            cout << "create dataset " << name
//                 << " with expected " << expectedType << " at " << expectedTo << " through "
//                 << expectedFrom
//                 << " with given " << givenType << " at " << givenTo << " through "
//                 << givenFrom
//                 << " using " << location
//                 << endl;
            return make_shared<ExecutionResult>(false, true, "Created.");
        }

    private:
        string name_;
        string location_;
        vector<ColumnGroup> column_groups_;
        bool skip_header_;
    };

    class CodeBlock : public ExecutableStatement {
    public:
        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr<ExecutionResult> lastResult = make_shared<ExecutionResult>();
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

        void addChild(const shared_ptr<ExecutableStatement> &child) {
            children.push_back(child);
        }

    private:
        vector<shared_ptr<ExecutableStatement>> children{};
    };
}
#endif //HAPPYML_STATEMENTS_HPP
