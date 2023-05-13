//
// Created by Erik Hyrkas on 5/5/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_EXIT_STATEMENT_HPP
#define HAPPYML_EXIT_STATEMENT_HPP

#include "../execution_context.hpp"
#include <iostream>

namespace happyml {
    class ExitStatement : public ExecutableStatement {
    public:
        shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            cout << "Exiting..." << endl;
            return make_shared<ExecutionResult>(true);
        }
    };

}

#endif //HAPPYML_EXIT_STATEMENT_HPP
