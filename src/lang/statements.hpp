//
// Created by erikh on 3/25/2023.
//

#ifndef HAPPYML_STATEMENTS_HPP
#define HAPPYML_STATEMENTS_HPP

#include <string>
#include <iostream>
#include <utility>
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
                cout << "Available commands: " << endl;
                cout << "  create dataset <name> \n"
                        "  [with config <key> <value>] \n"
                        "  [with expected [<scalar|category|pixel>] at <column> [through <column>] ]\n"
                        "  [with given [<scalar|category|pixel>] at <column> [through <column>] ]\n"
                        "  using <local file or folder|url>\n"
                     << endl << endl;
                cout << "  list datasets [<starting with x>]" << endl << endl;

            } else if (help_menu_item_ == "task" || help_menu_item_ == "tasks") {
                cout << "Available commands: " << endl;
                cout << "  create task <task type> <task name> \n"
                        "  [with config <key> <value>]\n"
                        "  using <dataset name>"
                     << endl << endl;
                cout << "  execute task <task name> \n"
                        "  [with label <label>] \n"
                        "  [with config <key> <value>] \n"
                        "  using dataset <dataset>" << endl << endl;
                cout << "  list tasks [<starting with x>]" << endl << endl;
                cout << "  refine task <task name> \n"
                        "  [with label [label]] \n"
                        "  [with config <key> <value>] \n"
                        "  using dataset <dataset name>" << endl << endl;

            } else if (help_menu_item_ == "future") {
                cout << "Future commands: " << endl;
                cout << "  copy <task name> [<label>] to [<task name>] [<label>]" << endl << endl;
                cout << "  copy <dataset name> to [<dataset name>]" << endl << endl;
                cout << "  delete <task name> [<label>]" << endl << endl;
                cout << "  delete <dataset name>" << endl << endl;

                cout << "  execute task <task name> \n"
                        "  [with label <label>] \n"
                        "  [with config <key> <value>] \n"
                        "  using input <csv encoded row>" << endl << endl;
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
        CreateDatasetStatement(string name, string location, string fileFormat,
                               string expectedType, size_t expectedTo, size_t expectedFrom,
                               string givenType, size_t givenTo, size_t givenFrom) :
                name(std::move(name)),
                location(std::move(location)),
                fileFormat(std::move(fileFormat)),
                expectedType(std::move(expectedType)),
                expectedFrom(expectedFrom),
                expectedTo(expectedTo),
                givenType(std::move(givenType)),
                givenFrom(givenFrom),
                givenTo(givenTo) {
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr<ExecutionResult> lastResult = make_shared<ExecutionResult>();
            // TODO: create dataset
            cout << "create dataset " << name << " from " << location
                 << " with format " << fileFormat
                 << " with expected " << expectedType << " at " << expectedTo << " through "
                 << expectedFrom
                 << " with given " << givenType << " at " << givenTo << " through "
                 << givenFrom
                 << endl;
            return lastResult;
        }

    private:
        string name;
        string location;
        string fileFormat;
        string expectedType;
        size_t expectedFrom;
        size_t expectedTo;
        string givenType;
        size_t givenFrom;
        size_t givenTo;
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
        vector<shared_ptr<ExecutableStatement>> children;
    };
}
#endif //HAPPYML_STATEMENTS_HPP
