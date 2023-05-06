//
// Created by Erik Hyrkas on 12/29/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TOKEN_HPP
#define HAPPYML_TOKEN_HPP

#include <string>
#include <vector>
#include <memory>
#include <sstream>

using namespace std;

namespace happyml {

    class Token {
    public:
        Token(size_t length, const string &label, const string &value, bool skip,
              size_t offset, const string &source) {
            this->length = length;
            this->label = label;
            this->value = value;
            this->skip = skip;
            this->offset = offset;
            this->source = source;
            this->lineIndex = 0; // right now, the lexer isn't counting lines. It should but that'll be for another day.
        }

        [[nodiscard]] size_t getLength() const {
            return length;
        }

        string getLabel() {
            return label;
        }

        string getValue() {
            return value;
        }

        [[nodiscard]] bool isSkip() const {
            return skip;
        }

        [[nodiscard]] size_t getOffset() const {
            return offset;
        }

        string getSource() {
            return source;
        }

        [[nodiscard]] size_t getLineIndex() const {
            return lineIndex;
        }

        string render() {
            stringstream result;
            string val = getValue();
            if (val == "\n") {
                val = "<\\n>";
            } else if (val == "\r") {
                val = "<\\r>";
            } else if (val == "\t") {
                val = "<\\t>";
            } else if (val == " ") {
                val = "<space>";
            }
            result << "[" << val << " (" << getSource() << ":" << getOffset() << ":" << getLabel() << ")]";
            return result.str();
        }

    private:
        size_t length;
        string label;
        string value;
        bool skip;
        size_t lineIndex;
        size_t offset;
        string source;
    };

    class TokenStream {
    public:
        explicit TokenStream(const vector<shared_ptr<Token>> &matches) {
            this->matches = matches;
            this->offset = 0;
        }

        [[nodiscard]] bool hasNext(size_t count = 1) const {
            size_t next_offset = offset + count - 1;
            return next_offset < matches.size();
        }

        [[nodiscard]] shared_ptr<Token> peek(size_t count = 1) const {
            if (!hasNext(count)) {
                throw std::out_of_range("Offset is out of range.");
            }
            size_t next_offset = offset + count - 1;
            return matches[next_offset];
        }

        shared_ptr<Token> previous() {
            if (offset == 0) {
                return nullptr;
            }
            return matches[offset - 1];
        }

        shared_ptr<Token> next() {
            auto result = peek();
            consume();
            return result;
        }

        void consume(size_t count = 1) {
            if (!hasNext(count)) {
                throw std::out_of_range("Offset is out of range.");
            }
            offset += count;
        }

        string render() {
            stringstream result;
            for (const auto &match: matches) {
                result << match->render() << endl;
            }
            return result.str();
        }

        size_t size() {
            return matches.size();
        }

    private:
        vector<shared_ptr<Token>> matches;
        size_t offset;
    };

}
#endif //HAPPYML_TOKEN_HPP
