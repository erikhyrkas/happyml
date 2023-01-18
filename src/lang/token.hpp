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

    private:
        size_t length;
        string label;
        string value;
        bool skip;
        size_t lineIndex;
        size_t offset;
        string source;
    };

    // There is likely a more C++ focused approach around iterators
    // but my C++ is rusty, so I'll do what I know I can make work,
    // though this probably can be refined later.
    class MatchStream {
    public:
        MatchStream(const vector<shared_ptr<Match>> &matches) {
            this->matches = matches;
        }

        bool hasNext() {
            if (matches.empty() || offset >= matches.size()) {
                return false;
            }
            size_t start = fresh ? 0 : offset + 1;
        }

        string render() {
            stringstream result;
            for (const auto match: matches) {
                string val = match->getValue();
                if (val == "\n") {
                    val = "<\\n>";
                } else if (val == "\r") {
                    val = "<\\r>";
                } else if (val == "\t") {
                    val = "<\\t>";
                } else if (val == " ") {
                    val = "<space>";
                }
                result << val << "\t[" << match->getSource() << ":" << match->getOffset() << ":" << match->getLabel()
                       << "]" << endl;
            }
            return result.str();
        }

        size_t size() {
            return matches.size();
        }

    private:
        vector<shared_ptr<Match>> matches;
        size_t offset;
        bool fresh;
        size_t lastConsumedOffset;

    };

}
#endif //HAPPYML_TOKEN_HPP
