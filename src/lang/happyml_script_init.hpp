//
// Created by Erik Hyrkas on 1/18/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HAPPYML_SCRIPT_INIT_HPP
#define HAPPYML_HAPPYML_SCRIPT_INIT_HPP

#include "lexer.hpp"
#include "parser.hpp"

using namespace std;
using namespace happyml;

namespace happyml {
    shared_ptr<Lexer> initializeHappymlLexer() {
        // I might want to reuse the lexer to parse other things
        // than our scripting language. So, we pass in the patterns.

        vector<shared_ptr<Pattern>> patterns;

        patterns.push_back(createKeywordToken("at"));
        patterns.push_back(createKeywordToken("config"));
        patterns.push_back(createKeywordToken("copy"));
        patterns.push_back(createKeywordToken("create"));
        patterns.push_back(createKeywordToken("dataset"));
        patterns.push_back(createKeywordToken("datasets"));
        patterns.push_back(createKeywordToken("delete"));
        patterns.push_back(createKeywordToken("execute"));
        patterns.push_back(createKeywordToken("expected"));
        patterns.push_back(createKeywordToken("exit"));
        patterns.push_back(createKeywordToken("given"));
        patterns.push_back(createKeywordToken("help"));
        patterns.push_back(createKeywordToken("input"));
        patterns.push_back(createKeywordToken("label"));
        patterns.push_back(createKeywordToken("limit"));
        patterns.push_back(createKeywordToken("list"));
        patterns.push_back(createKeywordToken("move"));
        patterns.push_back(createKeywordToken("pixel"));
        patterns.push_back(createKeywordToken("print"));
        patterns.push_back(createKeywordToken("refine"));
        patterns.push_back(createKeywordToken("scalar"));
        patterns.push_back(createKeywordToken("task"));
        patterns.push_back(createKeywordToken("tasks"));
        patterns.push_back(createKeywordToken("to"));
        patterns.push_back(createKeywordToken("using"));
        patterns.push_back(createKeywordToken("value"));
        patterns.push_back(createKeywordToken("with"));

        patterns.push_back(createCommentPattern());
        patterns.push_back(createStringPattern());
        patterns.push_back(createNumberPattern());
        patterns.push_back(createWordPattern());

        // punctuation
        patterns.push_back(createToken("_open_parenthesis", "("));
        patterns.push_back(createToken("_close_parenthesis", ")"));
        patterns.push_back(createToken("_equal", "="));
        patterns.push_back(createToken("_colon", ":"));
        patterns.push_back(createToken("_slash", "/"));
        patterns.push_back(createToken("_dot", "."));
        patterns.push_back(createToken("_percent", "%"));
        patterns.push_back(createToken("_backslash", "\\"));
        patterns.push_back(createToken("_double_quote", "\""));
        patterns.push_back(createToken("_single_quote", "\'"));
        patterns.push_back(createToken("_comma", ","));
        patterns.push_back(createToken("_underscore", "_"));
        patterns.push_back(createSkippedToken("_newline", "\n"));
        patterns.push_back(createSkippedToken("_tab", "\t"));
        patterns.push_back(createSkippedToken("_return", "\r"));
        patterns.push_back(createSkippedToken("_space", " "));
        auto lexer = make_shared<Lexer>(patterns);

        return lexer;
    }

    shared_ptr<Parser> initializeHappymlParser() {
        // load lexer and parser rules
        auto lexer = initializeHappymlLexer();
        auto parser = make_shared<Parser>(lexer);
        return parser;
    }


}
#endif //HAPPYML_HAPPYML_SCRIPT_INIT_HPP
