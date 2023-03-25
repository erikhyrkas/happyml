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
        // objects
        patterns.push_back(createKeywordToken("model"));
        patterns.push_back(createKeywordToken("dataset"));
        patterns.push_back(createKeywordToken("row"));
        patterns.push_back(createKeywordToken("rows"));
        patterns.push_back(createKeywordToken("column"));
        patterns.push_back(createKeywordToken("columns"));
        // actions
        patterns.push_back(createKeywordToken("exit"));
        patterns.push_back(createKeywordToken("create"));
        patterns.push_back(createKeywordToken("train"));
        patterns.push_back(createKeywordToken("retrain"));
        patterns.push_back(createKeywordToken("tune"));
        patterns.push_back(createKeywordToken("predict"));
        patterns.push_back(createKeywordToken("infer"));
        patterns.push_back(createKeywordToken("validate"));
        patterns.push_back(createKeywordToken("set"));
        patterns.push_back(createKeywordToken("let"));
        patterns.push_back(createKeywordToken("copy"));
        patterns.push_back(createKeywordToken("log"));
        // criteria
        patterns.push_back(createKeywordToken("with"));
        patterns.push_back(createKeywordToken("at"));
        patterns.push_back(createKeywordToken("from"));
        patterns.push_back(createKeywordToken("through"));
        patterns.push_back(createKeywordToken("add"));
        patterns.push_back(createKeywordToken("using"));
        patterns.push_back(createKeywordToken("given"));
//            I don't think these need to be keywords.
//            // adjectives
//            patterns.push_back(createKeyword("fast"));
//            patterns.push_back(createKeyword("small"));
//            patterns.push_back(createKeyword("accurate"));
//            patterns.push_back(createKeyword("comma"));
//            patterns.push_back(createKeyword("tab"));


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
        patterns.push_back(createToken("_backslash", "\\"));
        patterns.push_back(createToken("_double_quote", "\""));
        patterns.push_back(createToken("_single_quote", "\'"));
        patterns.push_back(createToken("_comma", ","));
        patterns.push_back(createToken("_newline", "\n"));
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
