//
// Created by Erik Hyrkas on 12/29/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TOKEN_HPP
#define HAPPYML_TOKEN_HPP

#include <string>
#include <vector>

using namespace std;

namespace happyml {

    class Match {
    public:
        Match(size_t length, const string &label, const string &value, bool skip,
              size_t offset, const string &source) {
            this->length = length;
            this->label = label;
            this->value = value;
            this->skip = skip;
            this->offset = offset;
            this->source = source;
        }

        size_t getLength() {
            return length;
        }

        string getLabel() {
            return label;
        }

        string getValue() {
            return value;
        }

        bool isSkip() {
            return skip;
        }

        size_t getOffset() {
            return offset;
        }

        string getSource() {
            return source;
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

    class MatchStream {
    public:
        explicit MatchStream(const vector<shared_ptr<Match>> &matches) {
            this->matches = matches;
            this->offset = 0;
        }

        [[nodiscard]] bool hasNext(size_t count=1) const {
            size_t next_offset = offset + count - 1;
            return next_offset < matches.size();
        }

        [[nodiscard]] shared_ptr<Match> peek(size_t count=1) const {
            if (!hasNext(count)) {
                throw std::out_of_range("Offset is out of range.");
            }
            size_t next_offset = offset + count - 1;
            return matches[next_offset];
        }

        shared_ptr<Match> previous() {
            if(offset == 0 ) {
                return nullptr;
            }
            return matches[offset-1];
        }

        shared_ptr<Match> next() {
            auto result = peek();
            consume();
            return result;
        }

        void consume(size_t count=1) {
            if (!hasNext(count)) {
                throw std::out_of_range("Offset is out of range.");
            }
            offset += count;
        }

        string render() {
            stringstream result;
            for (const auto match: matches) {
                result << match->render() << endl;
            }
            return result.str();
        }

        size_t size() {
            return matches.size();
        }

    private:
        vector<shared_ptr<Match>> matches;
        size_t offset;
    };

}
#endif //HAPPYML_TOKEN_HPP
