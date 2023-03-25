//
// Created by Erik Hyrkas on 3/24/2023.
//

#ifndef HAPPYML_HAPPYML_AST_NODES_HPP
#define HAPPYML_HAPPYML_AST_NODES_HPP

#include <string>
#include <utility>
#include <vector>
#include "token.hpp"

using namespace std;

namespace happyml {
    class AstExecutionResult {
    public:
        explicit AstExecutionResult(bool exit = false, bool success = true, const string &message = "") {
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

    class ExecutionContext {

    };

    class AstNode {
    public:
//        virtual bool matches(const shared_ptr<MatchStream> &stream) = 0;
    };

    class ExecutableAstNode : public AstNode {
    public:
        virtual shared_ptr<AstExecutionResult> execute(const shared_ptr<ExecutionContext> &context) = 0;
    };

    class AstParseResult {
    public:
        explicit AstParseResult(const string &message = "Failure", bool success = false) {
            this->success = success;
            this->message = message;
        }

        explicit AstParseResult(const shared_ptr<ExecutableAstNode> &node, const string &message = "Success",
                                bool success = true) {
            this->success = success;
            this->message = message;
            this->node = node;
        }

        [[nodiscard]] bool isSuccessful() const {
            return success;
        }

        string getMessage() {
            return message;
        }

        shared_ptr<ExecutableAstNode> getNode() {
            return node;
        }

    private:
        bool success;
        string message;
        shared_ptr<ExecutableAstNode> node;
    };


    class CreateDataset : public ExecutableAstNode {
    public:
        CreateDataset(string name, string location, string fileFormat,
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

        shared_ptr<AstExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr<AstExecutionResult> lastResult = make_shared<AstExecutionResult>();
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

    class CodeBlock : public ExecutableAstNode {
    public:
        shared_ptr<AstExecutionResult> execute(const shared_ptr<ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr<AstExecutionResult> lastResult = make_shared<AstExecutionResult>();
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

        void addChild(const shared_ptr<ExecutableAstNode> &child) {
            children.push_back(child);
        }

    private:
        vector<shared_ptr<ExecutableAstNode>> children;
    };


    shared_ptr<AstParseResult> generateError(const string &message, const shared_ptr<Match> &token) {
        stringstream error_message;
        error_message << message << token->render();
        return make_shared<AstParseResult>(error_message.str(), false);
    }

    int parseColumnValue(const shared_ptr<MatchStream> &stream) {
        try {
            return stoi(stream->next()->getValue());
        } catch (invalid_argument const &ex) {
            throw runtime_error("Invalid Value: " + stream->previous()->render());
        } catch (out_of_range const &ex) {
            throw runtime_error("Invalid Value: " + stream->previous()->render());
        }
    }

    int tryParseThroughRange(const shared_ptr<MatchStream> &stream) {
        if (!stream->hasNext() || "_through" != stream->peek()->getLabel()) {
            // we default to -1 if there was no value.
            return -1;
        }
        stream->consume();
        if (!stream->hasNext()) {
            throw runtime_error("Missing value: " + stream->previous()->render());
        }
        return parseColumnValue(stream);
    }

    shared_ptr<AstParseResult> parseCreateDataset(const shared_ptr<MatchStream> &stream,
                                                  shared_ptr<Match> &next) {
        //create dataset <name> from <local file|local folder|url>
        //[with format <delimited|image>]
        //[with expected [<scalar|category|pixel>] at <column> [through <column>] ]
        //[with given [<scalar|category|pixel>] at <column> [through <column>] ]
        if (!stream->hasNext()) {
            return generateError("create dataset requires a name: ", next);
        }
        auto datasetName = stream->next();
        string name = datasetName->getValue();
        if (!stream->hasNext(2)) {
            return generateError("create dataset requires a location: ", datasetName);
        }
        auto fromKeyword = stream->next();
        if ("_from" != fromKeyword->getLabel()) {
            return generateError("Invalid token at: ", fromKeyword);
        }
        auto datasetLocation = stream->next();
        string location = datasetLocation->getValue();
        string fileFormat = "csv";
        string expectedType = "scalar";
        int expectedFrom = 0;
        int expectedTo = -1;
        string givenType = "scalar";
        int givenFrom = 1;
        int givenTo = -1;

        try {
            while ("_with" == stream->peek()->getLabel()) {
                stream->consume();
                if (!stream->hasNext(4)) {
                    return generateError("with statement is malformed: ", stream->previous());
                }
                string withType = stream->next()->getValue();
                if ("expected" == withType) {
                    expectedType = stream->next()->getValue();
                    if ("_at" != stream->next()->getLabel()) {
                        return generateError("with statement is malformed: ", stream->previous());
                    }
                    expectedFrom = parseColumnValue(stream);
                    expectedTo = tryParseThroughRange(stream);
                } else if ("given" == withType) {
                    givenType = stream->next()->getValue();
                    if ("_at" != stream->next()->getLabel()) {
                        return generateError("with statement is malformed: ", stream->previous());
                    }
                    givenFrom = parseColumnValue(stream);
                    givenTo = tryParseThroughRange(stream);
                } else {
                    return generateError("with statement is malformed ", stream->previous());
                }
            }
        } catch (runtime_error &e) {
            return generateError(e.what(), stream->previous());
        }
        auto createDataset = make_shared<CreateDataset>(name, location, fileFormat,
                                                        expectedType, expectedTo, expectedFrom,
                                                        givenType, givenTo, givenFrom);
        return make_shared<AstParseResult>(createDataset);
    }

    shared_ptr<AstParseResult> parseCreateStatement(const shared_ptr<MatchStream> &stream) {
        if (!stream->hasNext()) {
            return generateError("Incomplete statement at: ", stream->previous());
        }
        auto next = stream->next();
        auto label = next->getLabel();
        if ("_dataset" == label) {
            return parseCreateDataset(stream, next);
        }
        return generateError("Unsupported object for create: ", next);
    }

    shared_ptr<AstParseResult> parseCodeBlock(const shared_ptr<MatchStream> &stream) {
        auto codeBlock = make_shared<CodeBlock>();
        shared_ptr<AstParseResult> result = make_shared<AstParseResult>(codeBlock);
        while (stream->hasNext()) {
            auto next = stream->next();
            string label = next->getLabel();
            if ("_create" == label) {
                auto createStatementResult = parseCreateStatement(stream);
                if (!createStatementResult->isSuccessful()) {
                    return createStatementResult;
                }
                codeBlock->addChild(createStatementResult->getNode());
            } else {
                return generateError("Unexpected token: ", next);
            }
        }

        return result;
    }
}
#endif //HAPPYML_HAPPYML_AST_NODES_HPP
