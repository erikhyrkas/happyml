//
// Created by Erik Hyrkas on 12/28/2022.
// Copyright 2022. Usable under MIT license.
//

// A parser takes in a list of tokens from a lexer and creates meaning from it.
// In this case, the goal is to create a dsl object that we can execute.
#ifndef HAPPYML_PARSER_HPP
#define HAPPYML_PARSER_HPP

#include "lexer.hpp"
#include "session_state.hpp"
#include "executable_script.hpp"

using namespace std;

namespace happyml {
    class ParseError : public ExecutableScript {
    public:
        ParseError(const string &errorMessage) {
            this->errorMessage = errorMessage;
        }

        bool execute(const shared_ptr<SessionState> &sessionState) override {
            // print error
            cerr << errorMessage << endl;
            return true;
        }

    private:
        string errorMessage;
    };

    // A stateless parser, responsible for building an executable script.
    // The results could be cached, so we don't want to allow any session
    // state to influence what we build.
    class Parser {
    public:
        explicit Parser(shared_ptr<Lexer> lexer) {
            this->lexer = lexer;
        }

        shared_ptr<ExecutableScript> parse(const string &text, const string &source = "unknown") {
            // lex to get tokens
            auto lexResult = lexer->lex(text, source);
            if (!lexResult->getMatchStream()) {
                return make_shared<ParseError>(lexResult->getMessage());
            }
            cout << "Lexer: " << lexResult->getMessage() << endl << lexResult->getMatchStream()->render() << endl;
            // then build parsed result.
            return make_shared<ParseError>("Parser is a work in progress.");
        }

    private:
        shared_ptr<Lexer> lexer;
    };

}
#endif //HAPPYML_PARSER_HPP
