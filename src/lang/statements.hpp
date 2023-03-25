//
// Created by erikh on 3/25/2023.
//

#ifndef HAPPYML_STATEMENTS_HPP
#define HAPPYML_STATEMENTS_HPP

#include <string>
#include <iostream>
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
