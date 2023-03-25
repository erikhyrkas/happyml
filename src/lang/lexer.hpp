//
// Created by Erik Hyrkas on 12/28/2022.
// Copyright 2022. Usable under MIT license.
//

// The purpose of a lexer is to take in text and turn it into tokens,
// those tokens are fed to the parser to produce meaning.
#ifndef HAPPYML_LEXER_HPP
#define HAPPYML_LEXER_HPP

#include "pattern.hpp"

using namespace std;

// NOTE: the lexer currently doesn't do look behind or look ahead. This keeps
// the logic simple, but requires careful consideration when writing rules.
// It's also worth noting that the lexer attempts the shortest possible valid
// match rather than greedily trying to match the maximum length of a pattern.
namespace happyml {

    class LexerResult {
    public:
        LexerResult(const shared_ptr<TokenStream> &matchStream, const string &message) {
            this->matchStream = matchStream;
            this->message = message;
        }

        shared_ptr<TokenStream> getMatchStream() {
            return matchStream;
        }

        string getMessage() {
            return message;
        }

    private:
        shared_ptr<TokenStream> matchStream;
        string message;
    };

    class Lexer {
    public:
        explicit Lexer(const vector<shared_ptr<Pattern>> &patterns) {
            this->patterns = patterns;
        }

        shared_ptr<LexerResult> lex(const string &text, const string &source = "unknown") {
            vector<shared_ptr<Token>> matches;
            size_t scanLimit = text.length();
            size_t offset = 0;
            while (offset < scanLimit) {
                auto longestMatch = findLongestMatch(text, offset, source);
                if (!longestMatch) {
                    // TODO: we have more information we can share, just needs to be added.
                    // This is bare bones, but enough for the moment.
                    size_t remaining = text.length() - offset;
                    string sub = text.substr(offset, std::min(remaining, (size_t) 10));
                    stringstream message;
                    message << "Syntax error at: " << source << "(" << offset << ") [" << sub << "]" << endl;
                    return make_shared<LexerResult>(nullptr, message.str());
                }
                if (!longestMatch->isSkip()) {
                    matches.push_back(longestMatch);
                }
                const size_t len = longestMatch->getLength();
                if (len == 0) {
                    size_t remaining = text.length() - offset;
                    string sub = text.substr(offset, std::min(remaining, (size_t) 10));
                    stringstream message;
                    message << "Syntax error at: " << source << "(" << offset << ") [" << sub << "]" << endl;
                    return make_shared<LexerResult>(nullptr, message.str());
                }
                offset += len;
            }

            return make_shared<LexerResult>(make_shared<TokenStream>(matches), "success");
        }

    private:
        vector<shared_ptr<Pattern>> patterns;

        shared_ptr<Token> findLongestMatch(const string &text, size_t offset, const string &source = "unknown") {
            shared_ptr<Token> longestMatch;
            for (const auto &pattern: patterns) {
                auto nextMatch = pattern->match(text, offset, source);
                if (nextMatch && (!longestMatch || nextMatch->getLength() > longestMatch->getLength())) {
                    longestMatch = nextMatch;
                }
            }
            return longestMatch;
        }
    };

}
#endif //HAPPYML_LEXER_HPP
