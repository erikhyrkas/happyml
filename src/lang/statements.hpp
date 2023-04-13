//
// Created by Erik Hyrkas on 3/25/2023.
//

#ifndef HAPPYML_STATEMENTS_HPP
#define HAPPYML_STATEMENTS_HPP

#include <string>
#include <iostream>
#include <utility>
#include <vector>
#include "execution_context.hpp"

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
                     << "  [with expected <label|number|text|image> at <column> [through <column>] ]*" << endl
                     << "  [with given <label|number|text|image> at <column> [through <column>] ]*" << endl
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


    struct ColumnGroup {
        string dataType;
        int startIndex;
        int endIndex;
    };

    class CreateDatasetStatement : public ExecutableStatement {
    public:
        CreateDatasetStatement(string name,
                               string location,
                               vector<ColumnGroup> expected,
                               vector<ColumnGroup> given) :
                name(std::move(name)),
                location(std::move(location)),
                expected_(std::move(expected)),
                given_(std::move(given)) {
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // default to success if there are no children.
            if (location.find("file://") != 0) {
                return make_shared<ExecutionResult>(false, false,
                                                    "create dataset only supports file:// location type at the moment.");
            }
            for(const auto &columnGroup : expected_) {
                if (columnGroup.dataType != "label" &&
                        columnGroup.dataType != "number" &&
                        columnGroup.dataType != "text" &&
                        columnGroup.dataType != "image") {
                    return make_shared<ExecutionResult>(false, false,
                                                        "create dataset's expected type must be one of: scalar, category, pixel, or text.");
                }
                // todo
            }
            for(const auto &columnGroup : given_) {
                if (columnGroup.dataType != "label" &&
                    columnGroup.dataType != "number" &&
                    columnGroup.dataType != "text" &&
                    columnGroup.dataType != "image") {
                    return make_shared<ExecutionResult>(false, false,
                                                        "create dataset's given type must be one of: scalar, category, pixel, or text.");
                }
                // todo
            }
            //  create dataset <name>
            //  [with expected <label|number|text|image> at <column> [through <column>] ]*
            //  [with given <label|number|text|image> at <column> [through <column>] ]*
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
        string name;
        string location;
        vector<ColumnGroup> expected_;
        vector<ColumnGroup> given_;
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
