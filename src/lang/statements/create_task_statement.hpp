//
// Created by Erik Hyrkas on 5/13/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CREATE_TASK_STATEMENT_HPP
#define HAPPYML_CREATE_TASK_STATEMENT_HPP

#include <iostream>
#include "../execution_context.hpp"
#include "../../util/task_utils.hpp"

namespace happyml {

    //   create task <task type> <task name>
    //  [with goal <speed|accuracy|memory>]
    //  using <dataset name>
    class CreateTaskStatement : public ExecutableStatement {
    public:
        CreateTaskStatement(string task_type, string task_name, string goal, string dataset_name, string test_dataset_name) :
                task_type_(std::move(task_type)),
                task_name_(std::move(task_name)),
                goal_(std::move(goal)),
                dataset_name_(std::move(dataset_name)),
                test_dataset_name_(std::move(test_dataset_name)) {

        }


        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // check if dataset exists
            if (!context->dataset_exists(dataset_name_)) {
                string message = "Dataset " + dataset_name_ + " does not exist.";
                return make_shared<ExecutionResult>(false, false, message);
            }
            if ("label" != task_type_) {
                string message = "Unsupported task type " + task_type_ + ".";
                return make_shared<ExecutionResult>(false, false, message);
            }
            string dataset_path = context->get_dataset_path(dataset_name_);
            string test_dataset_path = test_dataset_name_.empty() ? "" : context->get_dataset_path(test_dataset_name_);
            string task_folder_path = context->get_base_task_folder_path();
            if (!create_happyml_task(task_type_, task_name_,
                                     goal_, dataset_name_,
                                     dataset_path, task_folder_path,
                                     test_dataset_path)) {
                string message = "Failed to create task " + task_name_ + " of type " + task_type_ + " with goal " + goal_ + " using dataset " + dataset_name_;
                return make_shared<ExecutionResult>(false, false, message);
            }
            string success_message = "Created task " + task_name_ + " of type " + task_type_ + " with goal " + goal_ + " using dataset " + dataset_name_;

            return make_shared<ExecutionResult>(false, true, success_message);
        }

    private:
        string task_type_;
        string task_name_;
        string goal_;
        string dataset_name_;
        string test_dataset_name_;
    };
}
#endif //HAPPYML_CREATE_TASK_STATEMENT_HPP
