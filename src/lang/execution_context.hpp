//
// Created by Erik Hyrkas on 3/25/2023.
//

#ifndef HAPPYML_EXECUTION_CONTEXT_HPP
#define HAPPYML_EXECUTION_CONTEXT_HPP

#include <string>
#include <memory>

using namespace std;

namespace happyml {
    class ExecutionContext {
        // TODO: we'll eventually store state here.
        // Examples might include:
        // * external configuration
        // * session variables
        // * debugging/troubleshooting information
    };

    class ExecutionResult {
    public:
        explicit ExecutionResult(bool exit = false, bool success = true, const string &message = "") {
            this->success = success;
            this->message = message;
            this->exit = exit;
        }

        [[nodiscard]] bool exitRequested() const {
            return exit;
        }

        [[nodiscard]] bool isSuccessful() const {
            return success;
        }

        string getMessage() {
            return message;
        }

    private:
        bool success;
        bool exit;
        string message;
    };

    class ExecutableStatement {
    public:
        virtual shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) = 0;
    };

    class ParseResult {
    public:
        explicit ParseResult(const string &message = "Failure", bool success = false) {
            this->success = success;
            this->message = message;
        }

        explicit ParseResult(const shared_ptr<ExecutableStatement> &node, const string &message = "Success",
                             bool success = true) {
            this->success = success;
            this->message = message;
            this->executable = node;
        }

        [[nodiscard]] bool isSuccessful() const {
            return success;
        }

        string getMessage() {
            return message;
        }

        shared_ptr<ExecutableStatement> getExecutable() {
            return executable;
        }

    private:
        bool success;
        string message;
        shared_ptr<ExecutableStatement> executable;
    };
}

#endif //HAPPYML_EXECUTION_CONTEXT_HPP
