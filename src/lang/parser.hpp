//
// Created by Erik Hyrkas on 3/24/2023.
//

#ifndef HAPPYML_PARSER_HPP
#define HAPPYML_PARSER_HPP

#include <string>
#include <utility>
#include <vector>
#include "token.hpp"

using namespace std;

namespace happyml {
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

    class ExecutionContext {

    };

    class ExecutableAstNode {
    public:
        virtual shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) = 0;
    };

    class ParseResult {
    public:
        explicit ParseResult(const string &message = "Failure", bool success = false) {
            this->success = success;
            this->message = message;
        }

        explicit ParseResult(const shared_ptr <ExecutableAstNode> &node, const string &message = "Success",
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

        shared_ptr <ExecutableAstNode> getExecutable() {
            return node;
        }

    private:
        bool success;
        string message;
        shared_ptr <ExecutableAstNode> node;
    };

    class ExistStatement : public ExecutableAstNode {
    public:
        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            cout << "Exiting..." << endl;
            return make_shared<ExecutionResult>(true);
        }
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

        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr < ExecutionResult > lastResult = make_shared<ExecutionResult>();
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
        shared_ptr <ExecutionResult> execute(const shared_ptr <ExecutionContext> &context) override {
            // default to success if there are no children.
            shared_ptr < ExecutionResult > lastResult = make_shared<ExecutionResult>();
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

        void addChild(const shared_ptr <ExecutableAstNode> &child) {
            children.push_back(child);
        }

    private:
        vector<shared_ptr < ExecutableAstNode>> children;
    };

    class Parser {
    public:
        explicit Parser(const shared_ptr <Lexer> &lexer) : lexer(lexer) {
        }

        shared_ptr <ParseResult> parse(const string &text, const string &source = "unknown") {
            auto lexResult = lexer->lex(text, source);
            if (!lexResult->getMatchStream()) {
                return make_shared<ParseResult>(lexResult->getMessage(), false);
            }

            // cout << "Lexer: " << lexResult->getMessage() << endl << lexResult->getMatchStream()->render() << endl;
            // then build parsed result.
            return parseCodeBlock(lexResult->getMatchStream());
        }

    private:
        shared_ptr <Lexer> lexer;

        static shared_ptr <ParseResult> generateError(const string &message, const shared_ptr <Match> &token) {
            stringstream error_message;
            error_message << message << token->render();
            return make_shared<ParseResult>(error_message.str(), false);
        }

        static int parseColumnValue(const shared_ptr <MatchStream> &stream) {
            try {
                return stoi(stream->next()->getValue());
            } catch (invalid_argument const &ex) {
                throw runtime_error("Invalid Value: " + stream->previous()->render());
            } catch (out_of_range const &ex) {
                throw runtime_error("Invalid Value: " + stream->previous()->render());
            }
        }

        static int tryParseThroughRange(const shared_ptr <MatchStream> &stream) {
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

        static string parseLocation(const shared_ptr <MatchStream> &stream) {
            auto scheme = stream->next();
            if (!stream->hasNext(3)) {
                throw runtime_error("Malformed url at: " + scheme->render());
            }
            stream->consume(3); // :// that follows the https or file
            stringstream url;
            url << scheme->getValue() << "://";
            auto nextLabel = stream->peek()->getLabel();
            while ("_word" == nextLabel || "_slash" == nextLabel || "_backslash" == nextLabel
                   || "_dot" == nextLabel || "_colon" == nextLabel) {
                url << stream->next()->getValue();
                if(!stream->hasNext()) {
                    break;
                }
                nextLabel = stream->peek()->getLabel();
            }

            return url.str();
        }

        static shared_ptr <ParseResult> parseCreateDataset(const shared_ptr <MatchStream> &stream,
                                                           shared_ptr <Match> &next) {
            try {
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
                string location = parseLocation(stream);
                string fileFormat = "csv";
                string expectedType = "scalar";
                int expectedFrom = 0;
                int expectedTo = -1;
                string givenType = "scalar";
                int givenFrom = 1;
                int givenTo = -1;

                while (stream->hasNext() && "_with" == stream->peek()->getLabel()) {
                    stream->consume();
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
                    } else if ("format" == withType) {
                        fileFormat = stream->next()->getValue();
                    } else {
                        return generateError("with statement is malformed ", stream->previous());
                    }
                }
                auto createDataset = make_shared<CreateDataset>(name, location, fileFormat,
                                                                expectedType, expectedTo, expectedFrom,
                                                                givenType, givenTo, givenFrom);
                return make_shared<ParseResult>(createDataset);
            } catch (runtime_error &e) {
                return generateError(e.what(), stream->previous());
            }
        }

        static shared_ptr <ParseResult> parseCreateStatement(const shared_ptr <MatchStream> &stream) {
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

        static shared_ptr <ParseResult> parseCodeBlock(const shared_ptr <MatchStream> &stream) {
            auto codeBlock = make_shared<CodeBlock>();
            shared_ptr < ParseResult > result = make_shared<ParseResult>(codeBlock);
            while (stream->hasNext()) {
                auto next = stream->next();
                string label = next->getLabel();
                if ("_create" == label) {
                    auto createStatementResult = parseCreateStatement(stream);
                    if (!createStatementResult->isSuccessful()) {
                        return createStatementResult;
                    }
                    codeBlock->addChild(createStatementResult->getExecutable());
                } else if( "_exit" == label) {
                    codeBlock->addChild(make_shared<ExistStatement>());
                } else {
                    return generateError("Unexpected token: ", next);
                }
            }

            return result;
        }
    };
}
#endif //HAPPYML_PARSER_HPP
