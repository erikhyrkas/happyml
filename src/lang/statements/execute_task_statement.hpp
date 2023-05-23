#include <utility>

//
// Created by Erik Hyrkas on 5/23/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_EXECUTE_TASK_STATEMENT_HPP
#define HAPPYML_EXECUTE_TASK_STATEMENT_HPP

namespace happyml {
    class ExecuteTaskStatement : public ExecutableStatement {
    public:
        explicit ExecuteTaskStatement(string task_name, string task_label, string dataset_name,
                                      const unordered_map<string, string> &input_map) : task_name_(std::move(task_name)),
                                                                                        task_label_(std::move(task_label)),
                                                                                        dataset_name_(std::move(dataset_name)),
                                                                                        input_map_(input_map) {
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            //execute task <task name>
            //[with label <task label>]
            //using dataset <dataset name>
            //
            //      --or--
            //
            //execute task <task name>
            //[with label <task label>]
            //using input ("key": "value", "key": "value", ...)
            string task_folder_path = context->get_base_task_folder_path();
            string dataset_path = context->get_dataset_path(dataset_name_);
            // TODO: we ignore label for now. The label is hardcoded as "default" in a few spots and that needs to be refactored.
            // TODO: we need to cache the model so we don't have to reload it

            if (input_map_.empty()) {
                if (!execute_task_with_dataset(task_name_, dataset_path, task_folder_path)) {
                    string message = "Failed to execute task " + task_name_;
                    return make_shared<ExecutionResult>(false, false, message);
                }
            } else {
                // TODO: implement execute task with input
                return make_shared<ExecutionResult>(false, false, "Not implemented.");
            }

            return make_shared<ExecutionResult>(false, true, "Complete.");
        }

    private:
        string task_name_;
        string task_label_;
        string dataset_name_;
        unordered_map<string, string> input_map_;
    };
}
#endif //HAPPYML_EXECUTE_TASK_STATEMENT_HPP
