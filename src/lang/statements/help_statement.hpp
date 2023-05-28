//
// Created by Erik Hyrkas on 5/5/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HELP_STATEMENT_HPP
#define HAPPYML_HELP_STATEMENT_HPP

#include "../execution_context.hpp"
#include <utility>
#include <iostream>
#include <string>

namespace happyml {

    class HelpStatement : public ExecutableStatement {
    public:
        explicit HelpStatement(string help_menu_item = "default") : help_menu_item_(std::move(help_menu_item)) {

        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {

            if (help_menu_item_ == "dataset" || help_menu_item_ == "datasets") {
                cout << "Available dataset commands: " << endl;
                cout << "  create dataset <dataset name>" << endl;
                cout << "  [with header]" << endl;
                cout << "  [with given [label|number|text|image] <given name> [(<rows>, <columns>, <channels>)] at <column position>]+" << endl;
                cout << "  [with expected [label|number|text|image] <expected name> [(<rows>, <columns>, <channels>)] at <column position>]*" << endl;
                cout << "  using <file://path/>" << endl;
                cout << endl;
                cout << "  print {pretty|raw} <dataset name> [limit <limit number>]" << endl;
                cout << endl;
            } else if (help_menu_item_ == "task" || help_menu_item_ == "tasks") {
                cout << "Available task commands: " << endl;
                cout << "  create task {label} <task name>" << endl;
                cout << "  [with goal {speed|accuracy|memory}]" << endl;
                cout << "  [with test <test dataset name>]" << endl;
                cout << "  using <dataset name>" << endl;
            } else if (help_menu_item_ == "future") {
                cout << "Future general commands: " << endl;
                cout << "  set <property> <value>" << endl;
                cout << endl;
                cout << "  let <variable> <value>" << endl;
                cout << endl;
                cout << "Future dataset commands: " << endl;
                cout << "  describe dataset <dataset name>" << endl;
                cout << endl;
                cout << "  list datasets [starting with <start string>]" << endl;
                cout << endl;
                cout << "  copy dataset <source dataset name> to <destination dataset name>" << endl;
                cout << endl;
                cout << "  delete dataset <dataset name>" << endl;
                cout << endl;
                cout << "  move dataset <original dataset name> to <new dataset name>" << endl;
                cout << endl;
                cout << "  split dataset <dataset name> at <percent>" << endl;
                cout << endl;
                cout << "Future task commands: " << endl;
                cout << "  describe task <task name> [with label <task label>]" << endl;
                cout << endl;
                cout << "  list tasks [starting with <start string>]" << endl;
                cout << endl;
                cout << "  execute task <task name>" << endl;
                cout << "  [with label <task label>]" << endl;
                cout << "  using dataset <dataset name>" << endl;
                cout << endl;
                cout << "  execute task <task name>" << endl;
                cout << "  [with label <task label>]" << endl;
                cout << "  using input (key: \"value\", key: value,  key: [0,1,2], ...)" << endl;
                cout << endl;
                cout << "  refine task <task name>" << endl;
                cout << "  [with label <task label>]" << endl;
                cout << "  using dataset <dataset name>" << endl;
                cout << endl;
                cout << "  copy task <original task name> [with label <original task label>] to <new task name> [with label <new task label>]" << endl;
                cout << endl;
                cout << "  delete task <task name> [with label <task label>]" << endl;
                cout << endl;
                cout << "  move task <original task name> [with label <original task label>] to <new task name> [with label <new task label>]" << endl;
            } else {
                cout << "Available commands: " << endl;
                cout << "  exit" << endl;
                cout << endl;
                cout << "  help [{dataset|task|future}]" << endl;
            }

            return make_shared<ExecutionResult>(false);
        }

    private:
        string help_menu_item_;
    };
}
#endif //HAPPYML_HELP_STATEMENT_HPP
