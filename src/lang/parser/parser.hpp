//
// Created by Erik Hyrkas on 3/24/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_PARSER_HPP
#define HAPPYML_PARSER_HPP

#include <string>
#include <vector>
#include "../statements/code_block_statement.hpp"
#include "../statements/exit_statement.hpp"
#include "../statements/print_statement.hpp"
#include "../statements/help_statement.hpp"
#include "../statements/create_dataset_statement.hpp"
#include "../statements/create_task_statement.hpp"
#include "../statements/execute_task_statement.hpp"
#include "../happyml_variant.hpp"

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
            if (!stream->hasNext()) {
                return generateError("with statement data type missing ", stream->previous());
            }
            columnGroup->data_type_ = stream->next()->getValue();
            if (!stream->hasNext()) {
                return generateError("with statement label missing ", stream->previous());
            }
            columnGroup->label_ = stream->next()->getValue();
            auto dim_or_at = stream->next()->getLabel();
            if ("_open_parenthesis" == dim_or_at && stream->hasNext()) {
                auto next_val = parseNextNumber(stream);
                if (!stream->hasNext()) {
                    return generateError("with statement data type dimensions is incomplete: ", stream->previous());
                }
                auto next_token = stream->next()->getLabel();
                if ("_close_parenthesis" == next_token) {
                    columnGroup->rows_ = 1;
                    columnGroup->columns_ = next_val;
                    columnGroup->channels_ = 1;
                } else {
                    if ("_comma" != next_token || !stream->hasNext()) {
                        return generateError("with statement data type dimensions expected a comma after: ", stream->previous());
                    }
                    columnGroup->rows_ = next_val;
                    columnGroup->columns_ = parseNextNumber(stream);
                    next_token = stream->next()->getLabel();
                    if ("_close_parenthesis" == next_token) {
                        columnGroup->channels_ = 1;
                    } else {
                        if ("_comma" != next_token || !stream->hasNext()) {
                            return generateError("with statement data type dimensions expected a comma after: ", stream->previous());
                        }
                        columnGroup->channels_ = parseNextNumber(stream);
                        if ("_close_parenthesis" != stream->next()->getLabel() || !stream->hasNext()) {
                            return generateError("with statement data type dimensions expected a closing parenthesis after: ", stream->previous());
                        }
                    }
                }
                dim_or_at = stream->next()->getLabel();
            } else {
                columnGroup->rows_ = 1;
                columnGroup->columns_ = 1;
                columnGroup->channels_ = 1;
            }
            if ("_at" != dim_or_at || !stream->hasNext()) {
                return generateError("with statement expected \"at\" after: ", stream->previous());
            }
            columnGroup->start_index_ = parseNextNumber(stream);
            columnGroup->source_column_count_ = columnGroup->rows_ * columnGroup->columns_ * columnGroup->channels_;
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

        static shared_ptr<ParseResult> parseCreateTask(const shared_ptr<TokenStream> &stream,
                                                       shared_ptr<Token> &next) {
            try {
                // create task <task type> <task name>
                // [with goal <speed|accuracy|memory>]
                // [with test <dataset name>]
                // using <dataset name>
                if (!stream->hasNext()) {
                    return generateError("create task requires a type: ", next);
                }
                auto taskType = stream->next();
                if (!stream->hasNext()) {
                    return generateError("create task requires a name: ", next);
                }
                auto taskName = stream->next();
                if (!stream->hasNext()) {
                    return generateError("create task requires a dataset: ", next);
                }
                string goal = "accuracy";
                string test_dataset_name;
                while ("_with" == stream->peek()->getLabel()) {
                    auto withToken = stream->next();
                    if (!stream->hasNext()) {
                        return generateError("create task with statement malformed: ", next);
                    }
                    auto parameterToken = stream->next();
                    if ("test" == parameterToken->getValue()) {
                        if (!stream->hasNext()) {
                            return generateError("create task with statement malformed: ", parameterToken);
                        }
                        test_dataset_name = stream->next()->getValue();
                    } else if ("goal" == parameterToken->getValue()) {
                        if (!stream->hasNext()) {
                            return generateError("create task with statement malformed: ", parameterToken);
                        }
                        goal = stream->next()->getValue();
                    } else {
                        return generateError("create task with statement malformed: ", parameterToken);
                    }
                }
                if (!stream->hasNext()) {
                    return generateError("create task using statement malformed: ", next);
                }
                auto usingToken = stream->next();
                if ("_using" != usingToken->getLabel() || !stream->hasNext()) {
                    return generateError("create task using statement malformed: ", usingToken);
                }
                auto datasetName = stream->next();
                auto createTaskResult = make_shared<CreateTaskStatement>(taskType->getValue(),
                                                                         taskName->getValue(),
                                                                         goal,
                                                                         datasetName->getValue(),
                                                                         test_dataset_name);
                return make_shared<ParseResult>(createTaskResult);
            } catch (runtime_error &e) {
                return generateError(e.what(), stream->previous());
            }
        }

        static shared_ptr<ParseResult> parseCreateDataset(const shared_ptr<TokenStream> &stream,
                                                          shared_ptr<Token> &next) {
            try {
                //  create dataset <name>
                //  [with header]
                //  [with given <label|number|text|image> <name> [(<rows>, <columns>, <channels>)] at <column> ]+
                //  [with expected <label|number|text|image> <name> [(<rows>, <columns>, <channels>)] at <column> ]*
                //  using <file://path/>
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
                vector<shared_ptr<ColumnGroup >>
                        column_groups;
                bool has_header = false;
                while (stream->hasNext() && "_with" == stream->peek()->getLabel()) {
                    stream->consume();
                    auto columnGroup = make_shared<ColumnGroup>();
                    if (!stream->hasNext()) {
                        return generateError("with statement is incomplete ", stream->previous());
                    }
                    string withType = stream->next()->getValue();
                    if ("header" == withType) {
                        has_header = true;
                    } else {
                        if ("expected" == withType || "given" == withType) {
                            columnGroup->use_ = withType;
                        } else {
                            string message = "Unknown with type: " + withType;
                            return generateError(message, stream->previous());
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
            if ("_task" == label) {
                return parseCreateTask(stream, next);
            }
            return generateError("Unsupported object for create: ", next);
        }

        static unordered_map <std::string, std::vector<HappyMLVariant>> parseInput(const shared_ptr<TokenStream> &stream) {
            unordered_map<std::string, std::vector<HappyMLVariant>> inputs;
            if (!stream->hasNext()) {
                throw runtime_error("Missing input content");
            }
            // stream should have something like this in it: ("key": "value", "key": "value", ...)
            auto inputToken = stream->next();
            if ("_open_parenthesis" != inputToken->getLabel()) {
                throw runtime_error("Missing input content");
            }

            while (stream->hasNext() && "_close_parenthesis" != stream->peek()->getLabel()) {
                auto next_key = stream->next();
                if ("_word" != next_key->getLabel() && "_string" != next_key->getLabel()) {
                    throw runtime_error("Invalid input key");
                }
                string key = next_key->getValue();
                if ("_string" == next_key->getLabel()) {
                    key = unescapeString(key);
                }
                if (!stream->hasNext() || "_colon" != stream->peek()->getLabel()) {
                    throw runtime_error("Invalid input key");
                }
                stream->next(); // consume colon
                if (!stream->hasNext() || ("_string" != stream->peek()->getLabel() &&
                                           "_word" != stream->peek()->getLabel()) &&
                                          "_number" != stream->peek()->getLabel() &&
                                          "_open_bracket" != stream->peek()->getLabel()) {
                    throw runtime_error("Invalid input value");
                }
                std::vector<HappyMLVariant> next_value;
                if ("_open_bracket" == stream->peek()->getLabel()) {
                    stream->next(); // consume open bracket
                    while (stream->hasNext() && "_close_bracket" != stream->peek()->getLabel()) {
                        HappyMLVariant value;
                        auto token = stream->next();
                        string label = token->getLabel();
                        string data = token->getValue();
                        if ("_number" == label) {
                            value = stof(data);
                        } else if ("_string" == label) {
                            value = unescapeString(data);
                        } else {
                            value = data;
                        }
                        next_value.push_back(value);
                    }
                    if (!stream->hasNext()) {
                        throw runtime_error("Invalid input value");
                    }
                    stream->next(); // consume close bracket
                } else {
                    HappyMLVariant value;
                    auto token = stream->next();
                    string label = token->getLabel();
                    string data = token->getValue();
                    if ("_number" == label) {
                        value = stof(data);
                    } else if ("_string" == label) {
                        value = unescapeString(data);
                    } else {
                        value = data;
                    }
                    next_value.push_back(value);
                }
                // key needs to be lowercase
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                inputs[key] = next_value;
                if (stream->hasNext() && "_comma" == stream->peek()->getLabel()) {
                    stream->next(); // consume comma
                }
            }

            if (!stream->hasNext() || "_close_parenthesis" != stream->peek()->getLabel()) {
                throw runtime_error("Input incomplete, missing closing parenthesis");
            }
            stream->next(); // consume closing parenthesis

            return inputs;
        }

        static string &unescapeString(string &original) {// check for quote type
            string quote_type = original.substr(0, 1);
            original = original.substr(1, original.size() - 2);
            // unescape string. if we find a backslash followed by quote type, remove the backslash
            // otherwise, leave it alone
            for (int i = 0; i < original.size(); i++) {
                if (original[i] == '\\') {
                    if (i + 1 < original.size() && original[i + 1] == quote_type[0]) {
                        original.erase(i, 1);
                    }
                }
            }
            return original;
        }

        static shared_ptr<ParseResult> parseExecuteStatement(const shared_ptr<TokenStream> &stream) {
            try {
                //execute task <task name>
                //[with label <task label>]
                //using dataset <dataset name>
                //
                //      --or--
                //
                //execute task <task name>
                //[with label <task label>]
                //using input ("key": "value", "key": "value", ...)

                if (!stream->hasNext()) {
                    return generateError("execute requires a type: ", stream->previous());
                }
                auto executableType = stream->next(); // task
                if ("_task" != executableType->getLabel()) {
                    return generateError("task is the only valid executable right now: ", executableType);
                }
                if (!stream->hasNext()) {
                    return generateError("execute requires a name: ", stream->previous());
                }
                auto taskName = stream->next();
                if ("_word" != taskName->getLabel()) {
                    return generateError("task name is invalid: ", taskName);
                }
                string name = taskName->getValue();
                string label;
                if (stream->hasNext() && "_with" == stream->peek()->getLabel()) {
                    stream->consume();
                    if (!stream->hasNext()) {
                        return generateError("with statement is incomplete ", stream->previous());
                    }
                    auto withType = stream->next();
                    if ("_label" != withType->getLabel()) {
                        return generateError("with statement is invalid ", withType);
                    }
                    if (!stream->hasNext()) {
                        return generateError("with statement is incomplete ", stream->previous());
                    }
                    auto labelToken = stream->next();
                    if ("_word" != labelToken->getLabel()) {
                        return generateError("with statement is invalid ", labelToken);
                    }
                    label = labelToken->getValue();
                }
                if (!stream->hasNext()) {
                    return generateError("execute requires a dataset or input: ", stream->previous());
                }
                auto usingKeyword = stream->next();
                if ("_using" != usingKeyword->getLabel()) {
                    return generateError("Invalid token at: ", usingKeyword);
                }
                if (!stream->hasNext()) {
                    return generateError("execute requires a dataset or input: ", stream->previous());
                }
                auto next = stream->next();
                if ("_dataset" == next->getLabel()) {
                    if (!stream->hasNext()) {
                        return generateError("execute requires a dataset name: ", stream->previous());
                    }
                    auto datasetName = stream->next();
                    if ("_word" != datasetName->getLabel()) {
                        return generateError("dataset name is invalid: ", datasetName);
                    }
                    string dataset = datasetName->getValue();
                    std::unordered_map<std::string, std::vector<HappyMLVariant>> input_map;
                    auto executeTask = make_shared<ExecuteTaskStatement>(name, label, dataset, input_map);
                    return make_shared<ParseResult>(executeTask);
                } else if ("_input" == next->getLabel()) {
                    if (!stream->hasNext()) {
                        return generateError("execute requires an input: ", stream->previous());
                    }

                    std::unordered_map<std::string, std::vector<HappyMLVariant>> input_map = parseInput(stream);
                    auto executeTask = make_shared<ExecuteTaskStatement>(name, label, "", input_map);
                    return make_shared<ParseResult>(executeTask);
                } else {
                    return generateError("execute requires a dataset or input: ", next);
                }

            } catch (runtime_error &e) {
                return generateError(e.what(), stream->previous());
            }
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
                } else if ("_execute" == label) {
                    auto executeStatementResult = parseExecuteStatement(stream);
                    if (!executeStatementResult->isSuccessful()) {
                        return executeStatementResult;
                    }
                    codeBlock->addChild(executeStatementResult->getExecutable());
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
