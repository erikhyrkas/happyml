//
// Created by Erik Hyrkas on 12/30/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_EXECUTABLE_HPP
#define HAPPYML_EXECUTABLE_HPP

#include <memory>
#include "session_state.hpp"

using namespace std;

namespace happyml {
    class ExecutableResult {
    public:
        ExecutableResult(bool exit = false, bool success = true, const string &message = "") {
            this->success = success;
            this->message = message;
            this->exit = exit;
        }
        bool exitRequested() {
            return exit;
        }

        bool isSuccessful() {
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

    class Executable {
    public:
        virtual shared_ptr<ExecutableResult> execute(const shared_ptr<SessionState> &sessionState) = 0;
    };

    class ExecutableStatementBlock : public Executable {
    public:
        void addStatement(const shared_ptr<Executable> &executable) {
            statements.push_back(executable);
        }
        shared_ptr<ExecutableResult> execute(const shared_ptr<SessionState> &sessionState) override {
            // NOTE: could start a local state block if needed. Right now, everything is global
            for(const auto &statement : statements) {
                auto nextResult = statement->execute(sessionState);
                if(nextResult->exitRequested() || !nextResult->isSuccessful()) {
                    return nextResult;
                }
            }
            return make_shared<ExecutableResult>();
        }
    private:
        vector<shared_ptr<Executable>> statements;
    };

    class ParseErrorStatement : public Executable {
    public:
        ParseErrorStatement(const string &errorMessage) {
            this->errorMessage = errorMessage;
        }

        shared_ptr<ExecutableResult> execute(const shared_ptr<SessionState> &sessionState) override {
            return make_shared<ExecutableResult>(true, false, errorMessage);
        }

    private:
        string errorMessage;
    };

}
#endif //HAPPYML_EXECUTABLE_HPP
