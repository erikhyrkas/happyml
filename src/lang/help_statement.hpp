//
// Created by Erik Hyrkas on 5/5/2023.
//

#ifndef HAPPYML_HELP_STATEMENT_HPP
#define HAPPYML_HELP_STATEMENT_HPP

#include "execution_context.hpp"
#include <utility>
#include <iostream>
#include <string>

namespace happyml {

    class HelpStatement : public happyml::ExecutableStatement {
    public:
        explicit HelpStatement(string help_menu_item = "default") : help_menu_item_(std::move(help_menu_item)) {

        }

        shared_ptr<happyml::ExecutionResult> execute(const shared_ptr<happyml::ExecutionContext> &context) override {

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

                cout << "  create task <task type> <task name>" << endl
                     << "  [with goal <speed|accuracy|memory>]" << endl
                     << "  using <dataset name>" << endl << endl;

            } else if (help_menu_item_ == "future") {
                cout << "Future commands: " << endl;

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

            return make_shared<happyml::ExecutionResult>(false);
        }

    private:
        string help_menu_item_;
    };
}
#endif //HAPPYML_HELP_STATEMENT_HPP
