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
#include "../../util/pretty_print_row.hpp"

namespace happyml {

    class PrintStatement : public ExecutableStatement {
    public:
        explicit PrintStatement(string dataset_name, bool raw, int limit = -1) : dataset_name_(std::move(dataset_name)), raw_(raw), limit_(limit) {
        }

        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            string result_path = context->get_dataset_path(dataset_name_) + "/dataset.bin";
            BinaryDatasetReader reader(result_path);
            pretty_print(cout, reader, limit_, raw_);
            return make_shared<ExecutionResult>(false);
        }

    private:
        string dataset_name_;
        int limit_;
        bool raw_;
    };
}
#endif //HAPPYML_PRINT_STATEMENT_HPP
