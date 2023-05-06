//
// Created by Erik Hyrkas on 3/25/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_CODE_BLOCK_STATEMENT_HPP
#define HAPPYML_CODE_BLOCK_STATEMENT_HPP

#include <iostream>
#include <vector>
#include "execution_context.hpp"
#include "../training_data/training_dataset.hpp"

using namespace std;

namespace happyml {

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
#endif //HAPPYML_CODE_BLOCK_STATEMENT_HPP
