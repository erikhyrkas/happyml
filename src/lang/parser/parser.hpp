//
// Created by Erik Hyrkas on 3/24/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_PARSER_HPP
#define HAPPYML_PARSER_HPP

#include <string>
#include <vector>
#include "../lexer/token.hpp"
#include "../statements/code_block_statement.hpp"
#include "../statements/exit_statement.hpp"
#include "../statements/print_statement.hpp"
#include "../statements/help_statement.hpp"
#include "../statements/create_dataset_statement.hpp"

using namespace std;

namespace happyml {


    class Parser {
    public:
        explicit Parser(const shared_ptr<Lexer> &lexer) : lexer(lexer) {
        }

        shared_ptr<ParseResult> parse(const string &text, const string &source = "unknown") {
            auto lexResult = lexer->lex(text, source);
            if (!lexResult->getMatchStream()) {
                return make_shared<ParseResult>(lexResult->getMessage(), false);
            }

            // cout << "Lexer: " << lexResult->getMessage() << endl << lexResult->getMatchStream()->render() << endl;
            // then build parsed result.
            return parseCodeBlock(lexResult->getMatchStream());
        }

    private:
        shared_ptr<Lexer> lexer;

        static shared_ptr<ParseResult> generateError(const string &message, const shared_ptr<Token> &token) {
            stringstream error_message;
            error_message << message << token->render();
            return make_shared<ParseResult>(error_message.str(), false);
        }

        static shared_ptr<ParseResult> parseHelpStatement(const shared_ptr<TokenStream> &stream) {
            if (!stream->hasNext()) {
                return make_shared<ParseResult>(make_shared<HelpStatement>());
            }
            auto next = stream->next();
            return make_shared<ParseResult>(make_shared<HelpStatement>(next->getValue()));
        }

        static shared_ptr<ParseResult> parsePrintStatement(const shared_ptr<TokenStream> &stream) {
            if (!stream->hasNext()) {
                return generateError("usage: print <raw|pretty> <name> [limit <x>]", stream->previous());
            }
            auto next = stream->next();
            if ("raw" != next->getValue() && "pretty" != next->getValue()) {
                return generateError("usage: print <raw|pretty> <name> [limit <x>]", stream->previous());
            }
            bool raw = "raw" == next->getValue();
            next = stream->next();
            auto dataset_name = next->getValue();
            if (!stream->hasNext() || "_limit" != stream->peek()->getLabel()) {
                return make_shared<ParseResult>(make_shared<PrintStatement>(dataset_name, raw));
            }
            stream->next();
            auto limit = parseNextNumber(stream);
            return make_shared<ParseResult>(make_shared<PrintStatement>(next->getValue(), raw, limit));
        }

        static int parseNextNumber(const shared_ptr<TokenStream> &stream) {
            try {
                return stoi(stream->next()->getValue());
            } catch (invalid_argument const &ex) {
                throw runtime_error("Invalid Value: " + stream->previous()->render());
            } catch (out_of_range const &ex) {
                throw runtime_error("Invalid Value: " + stream->previous()->render());
            }
        }

        // It's important to note that this will take the rows, columns, channels
        // as best effort. However, labels will eventually be one-hot encoded and columns
        // will be updated. With text, it is encoded twice and then embedded, creating a whole
        // new shape. We still need this original shape, to know the user's intent.
        static shared_ptr<ParseResult> parseColumnGroup(shared_ptr<ColumnGroup> &columnGroup,
                                                        const shared_ptr<TokenStream> &stream) {
            columnGroup->data_type = stream->next()->getValue();
            auto dim_or_at = stream->next()->getLabel();
            if ("_open_parenthesis" == dim_or_at && stream->hasNext()) {
                auto next_val = parseNextNumber(stream);
                if (!stream->hasNext()) {
                    return generateError("with statement(1) is malformed ", stream->previous());
                }
                auto next_token = stream->next()->getLabel();
                if ("_close_parenthesis" == next_token) {
                    columnGroup->rows = 1;
                    columnGroup->columns = next_val;
                    columnGroup->channels = 1;
                } else {
                    if ("_comma" != next_token || !stream->hasNext()) {
                        return generateError("with statement(2) is malformed: ", stream->previous());
                    }
                    columnGroup->rows = next_val;
                    columnGroup->columns = parseNextNumber(stream);
                    next_token = stream->next()->getLabel();
                    if ("_close_parenthesis" == next_token) {
                        columnGroup->channels = 1;
                    } else {
                        if ("_comma" != next_token || !stream->hasNext()) {
                            return generateError("with statement(3) is malformed: ", stream->previous());
                        }
                        columnGroup->channels = parseNextNumber(stream);
                        if ("_close_parenthesis" != stream->next()->getLabel() || !stream->hasNext()) {
                            return generateError("with statement(4) is malformed: ", stream->previous());
                        }
                    }
                }
                dim_or_at = stream->next()->getLabel();
            } else {
                columnGroup->rows = 1;
                columnGroup->columns = 1;
                columnGroup->channels = 1;
            }
            if ("_at" != dim_or_at || !stream->hasNext()) {
                return generateError("with statement(5) is malformed: ", stream->previous());
            }
            columnGroup->start_index = parseNextNumber(stream);
            columnGroup->source_column_count = columnGroup->rows * columnGroup->columns * columnGroup->channels;
            return make_shared<ParseResult>("Success", true);
        }

        static string parseLocation(const shared_ptr<TokenStream> &stream) {
            auto scheme = stream->next();
            if (!stream->hasNext(3)) {
                throw runtime_error("Malformed url at: " + scheme->render());
            }
            stream->consume(3); // :// that follows the https or file
            stringstream url;
            url << scheme->getValue() << "://";
            auto nextLabel = stream->peek()->getLabel();
            while ("_word" == nextLabel || "_slash" == nextLabel || "_backslash" == nextLabel
                   || "_dot" == nextLabel || "_colon" == nextLabel || "_number" == nextLabel
                   || "_underscore" == nextLabel) {
                url << stream->next()->getValue();
                if (!stream->hasNext()) {
                    break;
                }
                nextLabel = stream->peek()->getLabel();
            }

            return url.str();
        }

        static shared_ptr<ParseResult> parseCreateDataset(const shared_ptr<TokenStream> &stream,
                                                          shared_ptr<Token> &next) {
            try {
                //  create dataset <name>
                //  [with header]
                //  [with given [<label|number|text|image>] at <column> [through <column>] ]+
                //  [with expected [<label|number|text|image>] at <column> [through <column>] ]*
                //  using <local file or folder|url>
                if (!stream->hasNext()) {
                    return generateError("create dataset requires a name: ", next);
                }
                auto datasetName = stream->next();
                string name = datasetName->getValue();
                if (datasetName->getLabel() != "_word") {
                    return generateError("create dataset name is invalid: ", datasetName);
                }
                if (!stream->hasNext(2)) {
                    return generateError("create dataset requires a location: ", datasetName);
                }
                vector<shared_ptr<ColumnGroup>> column_groups;
                bool has_header = false;
                while (stream->hasNext() && "_with" == stream->peek()->getLabel()) {
                    stream->consume();
                    auto columnGroup = make_shared<ColumnGroup>();
                    string withType = stream->next()->getValue();
                    if ("header" == withType) {
                        has_header = true;
                    } else {
                        if ("expected" == withType) {
                            columnGroup->use = withType;
                        } else if ("given" == withType) {
                            columnGroup->use = withType;
                        } else {
                            return generateError("with statement (0) is malformed ", stream->previous());
                        }
                        columnGroup->id_ = column_groups.size() + 1;
                        auto columnGroupParseResult = parseColumnGroup(columnGroup, stream);
                        if (!columnGroupParseResult->isSuccessful()) {
                            return columnGroupParseResult;
                        }
                        column_groups.push_back(columnGroup);
                    }
                }
                if (!stream->hasNext()) {
                    return generateError("missing using statement after: ", stream->previous());
                }
                auto usingKeyword = stream->next();
                if ("_using" != usingKeyword->getLabel()) {
                    return generateError("Invalid token at: ", usingKeyword);
                }
                string location = parseLocation(stream);

                auto createDataset = make_shared<CreateDatasetStatement>(name, location, has_header, column_groups);
                return make_shared<ParseResult>(createDataset);
//                return generateError("Unimplemented: Work in progress.", usingKeyword);
            } catch (runtime_error &e) {
                return generateError(e.what(), stream->previous());
            }
        }

        static shared_ptr<ParseResult> parseCreateStatement(const shared_ptr<TokenStream> &stream) {
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

        static shared_ptr<ParseResult> parseCodeBlock(const shared_ptr<TokenStream> &stream) {
            auto codeBlock = make_shared<CodeBlock>();
            shared_ptr<ParseResult> result = make_shared<ParseResult>(codeBlock);
            while (stream->hasNext()) {
                auto next = stream->next();
                string label = next->getLabel();
                if ("_newline" == next->getLabel()) {
                    continue;
                } else if ("_help" == label) {
                    auto helpStatementResult = parseHelpStatement(stream);
                    if (!helpStatementResult->isSuccessful()) {
                        return helpStatementResult;
                    }
                    codeBlock->addChild(helpStatementResult->getExecutable());
                } else if ("_print" == label) {
                    auto printStatementResult = parsePrintStatement(stream);
                    if (!printStatementResult->isSuccessful()) {
                        return printStatementResult;
                    }
                    codeBlock->addChild(printStatementResult->getExecutable());
                } else if ("_create" == label) {
                    auto createStatementResult = parseCreateStatement(stream);
                    if (!createStatementResult->isSuccessful()) {
                        return createStatementResult;
                    }
                    codeBlock->addChild(createStatementResult->getExecutable());
                } else if ("_exit" == label) {
                    codeBlock->addChild(make_shared<ExitStatement>());
                } else {
                    return generateError("Unexpected token: ", next);
                }
            }

            return result;
        }
    };
}
#endif //HAPPYML_PARSER_HPP
