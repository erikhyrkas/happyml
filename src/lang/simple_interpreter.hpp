//
// Created by Erik Hyrkas on 3/23/2023.
//

#ifndef HAPPYML_SIMPLE_INTERPRETER_HPP
#define HAPPYML_SIMPLE_INTERPRETER_HPP

#include <iostream>
#include <string>
#include <vector>

using namespace std;


namespace happyml {
    void simple_create_dataset(string &name, string &location, string &format,
                               string &expected_type, int expected_start_column_number, int expected_end_column_number,
                               string &give_type, int given_start_column_number, int given_end_column_number) {
        /*
         * create dataset <name> from <location>
         * [with format <format>]
         * [with expected <type> at <startColumnNumber> [through <endColumnNumber>] ]
         * [with given <type> at <startColumnNumber> [through <endColumnNumber>] ]
         */
        cout << "create dataset " << name << " from " << location
             << " with format " << format
             << " with expected " << expected_type << " at " << expected_start_column_number << " through "
             << expected_end_column_number
             << " with given " << give_type << " at " << given_start_column_number << " through "
             << given_end_column_number
             << endl;
    }

    void simple_train(vector<string> &adjectives, string &model_type, string &knowledge_label, string &dataset_name) {

    }

    void simple_predict(string &model_name, string &model_version, string &input) {

    }


// Define token types
    enum TokenType {
        IDENTIFIER,
        KEYWORD,
        STRING,
        INTEGER,
        FLOAT,
        COMMA,
        OPEN_BRACKET,
        CLOSE_BRACKET,
        END_OF_LINE
    };

// Define a token structure
    struct Token {
        TokenType type;
        string value;
        int line_number;
    };

// Define a lexer function to tokenize the input string
    vector<Token> simple_lexer(string input) {
        vector<Token> tokens;
        string current_token;
        int line_number = 1;
        for (int i = 0; i < input.length(); i++) {
            char current_char = input[i];
            if (current_char == ' ' || current_char == '\t') {
                // Ignore whitespace
                continue;
            } else if (current_char == ',') { // Add comma token
                tokens.push_back({COMMA, ",", line_number});
            } else if (current_char == '(') { // Add open bracket token
                tokens.push_back({OPEN_BRACKET, "(", line_number});
            } else if (current_char == ')') { // Add close bracket token
                tokens.push_back({CLOSE_BRACKET, ")", line_number});
            } else if (current_char == '\n') { // Add end of line token
                tokens.push_back({END_OF_LINE, "\n", line_number});
                line_number++;
            } else if (isdigit(current_char) || current_char == '.') { // Numeric value
                bool is_float = false;
                current_token += current_char; // Check if the value is a float
                for (int j = i + 1; j < input.length(); j++) {
                    char next_char = input[j];
                    if (isdigit(next_char)) {
                        current_token += next_char;
                    } else if (next_char == '.') {
                        current_token += next_char;
                        is_float = true;
                    } else {
                        break;
                    }
                    i++;
                } // Add integer or float token
                if (is_float) {
                    tokens.push_back({FLOAT, current_token, line_number});
                } else {
                    tokens.push_back({INTEGER, current_token, line_number});
                }
                current_token = "";
            } else if (current_char == '\"') {
                // Quoted string value
                current_token += current_char;
                for (int j = i + 1; j < input.length(); j++) {
                    char next_char = input[j];
                    if (next_char == '\"') {
                        current_token += next_char;
                        break;
                    }
                    current_token += next_char;
                    i++;
                } // Add string token
                tokens.push_back({STRING, current_token, line_number});
                current_token = "";
            } else { // Identifier or keyword
                current_token += current_char;
                for (int j = i + 1; j < input.length(); j++) {
                    char next_char = input[j];
                    // TODO: a url will not parse right since we don't count colon as part of a string,
                    //  but we'll need to fix that later since we wouldn't want random colons grouped with
                    //  keywords.
                    if (isalnum(next_char) || next_char == '_' || next_char == '/') {
                        current_token += next_char;
                    } else {
                        break;
                    }
                    i++;
                }
                // Check if the current token is a keyword
                if (current_token == "create" || current_token == "dataset" || current_token == "from" ||
                    current_token == "with" ||
                    current_token == "expected" || current_token == "given" || current_token == "through" ||
                    current_token == "simple_train" || current_token == "using" || current_token == "model" ||
                    current_token == "simple_predict") {
                    tokens.push_back({KEYWORD, current_token, line_number});
                } else { // Add identifier
                    tokens.push_back({IDENTIFIER, current_token, line_number});
                }
                current_token = "";
            }
        }
        return tokens;
    }


// Define a parse function to convert tokens into an abstract syntax tree (AST)
    bool simple_parse(const vector<Token>& tokens) {
        if (tokens.empty()) {
            return false;
        }
        vector<vector<Token >> commands;
        vector<Token> current_command;
        for (const Token& token: tokens) {
            if (token.type == END_OF_LINE) {
                commands.push_back(current_command);
                current_command.clear();
            } else {
                current_command.push_back(token);
            }
        }
        if (!current_command.empty()) {
            commands.push_back(current_command);
        }
        for (vector<Token> command: commands) {
            if (command[0].type == KEYWORD && command[0].value == "create") { // Create dataset command
                string name;
                string create_type;
                string location;
                string format;
                string expected_type;
                string given_type;
                int expected_start_column_number = 0;
                int expected_end_column_number = 0;
                int given_start_column_number = 0;
                int given_end_column_number = 0;
                for (int i = 1; i < command.size(); i++) {
                    Token token = command[i];
                    if (token.type == KEYWORD && i == 1) {
                        create_type = token.value;
                    } else if (token.type == IDENTIFIER && i == 2) {
                        name = token.value;
                    } else if (token.type == KEYWORD && token.value == "from") {
                        i++;
                        location = command[i].value;
                    } else if (token.type == KEYWORD && token.value == "with") {
                        i++;
                        if (command[i].value == "format") {
                            i++;
                            format = command[i].value;
                        } else if (command[i].value == "expected") {
                            expected_type = command[i + 1].value;
                            expected_start_column_number = stoi(command[i + 3].value);
                            if (command[i + 4].value == "through") {
                                expected_end_column_number = stoi(command[i + 5].value);
                                i += 5;
                            } else {
                                expected_end_column_number = expected_start_column_number;
                                i += 3;
                            }
                        } else if (command[i].value == "given") {
                            given_type = command[i + 1].value;
                            given_start_column_number = stoi(command[i + 3].value);
                            if (command[i + 4].value == "through") {
                                given_end_column_number = stoi(command[i + 5].value);
                                i += 5;
                            } else {
                                given_end_column_number = given_start_column_number;
                                i += 3;
                            }
                        }
                    }
                }
                if (create_type == "dataset") {
                    simple_create_dataset(name, location, format,
                                          expected_type, expected_start_column_number, expected_end_column_number,
                                          given_type, given_start_column_number, given_end_column_number);
                } else {
                    cout << "Invalid create type: " << create_type << endl;
                    return false;
                }
            } else if (command[0].type == KEYWORD && command[0].value == "simple_train") { // Train model command
                vector<string> adjectives;
                string model_type;
                string model_name;
                string knowledge_label;
                string dataset_name;
                for (int i = 1; i < command.size(); i++) {
                    Token token = command[i];
                    if (token.type == IDENTIFIER && i == 1) {
                        if (token.value == "a" || token.value == "an") {
                            i++;
                            adjectives.push_back(token.value);
                        }
                    } else if (token.type == IDENTIFIER && i == 2) {
                        model_type = token.value;
                    } else if (token.type == IDENTIFIER && i == 3) {
                        model_name = token.value;
                    } else if (token.type == IDENTIFIER && i == 4) {
                        if (token.value == "with") {
                            i++;
                            knowledge_label = command[i].value;
                        }
                    } else if (token.type == KEYWORD && token.value == "using") {
                        i++;
                        dataset_name = command[i].value;
                    }
                }
                simple_train(adjectives, model_type, knowledge_label, dataset_name);
            } else if (command[0].type == KEYWORD && command[0].value == "simple_predict") { // Predict command
                string model_name;
                string model_version;
                string input;
                for (int i = 1; i < command.size(); i++) {
                    Token token = command[i];
                    if (token.type == KEYWORD && token.value == "using") {
                        i++;
                        model_name = command[i].value;
                        if (command[i + 1].type == STRING) {
                            i += 1;
                            model_version = command[i].value;
                        }
                    } else if (token.type == KEYWORD && token.value == "given") {
                        i++;
                        input = command[i].value;
                    }
                }
                simple_predict(model_name, model_version, input);
            } else { // Invalid command
                cout << "Invalid command: ";
                for (const Token& token: command) {
                    cout << token.value << " ";
                }
                cout << endl;
                return false;
            }
        }
        return true;
    }

    bool simple_interpret(string &input) {
        // Tokenize the input string
        vector<Token> tokens = simple_lexer(input);
        // Parse the tokens and execute the commands
        return simple_parse(tokens);
    }
}
#endif //HAPPYML_SIMPLE_INTERPRETER_HPP
