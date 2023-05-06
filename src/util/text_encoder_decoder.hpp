//
// Created by Erik Hyrkas on 4/23/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TEXT_ENCODER_DECODER_HPP
#define HAPPYML_TEXT_ENCODER_DECODER_HPP

#include <sstream>
#include <string>

using namespace std;

namespace happyml {

    class TextEncoderDecoder {
    public:
        static string encodeString(const string &str, const char delimiter) {
            stringstream encoded;
            for (char c: str) {
                if (c == '<') {
                    encoded << "<_<";
                } else if (c == delimiter) {
                    encoded << "<~D>";
                } else if (c == '\n') {
                    encoded << "<~N>";
                } else {
                    encoded << c;
                }
            }
            return encoded.str();
        }

        static string decodeString(const string &str, const char delimiter) {
            stringstream decoded;
            for (size_t i = 0; i < str.size(); ++i) {
                if (str[i] == '<' && i + 2 < str.size()) {
                    ++i;
                    if (str[i] == '_') {
                        ++i;
                        if (str[i] == '<') {
                            decoded << '<';
                        } else if (str[i] == '_') {
                            decoded << "<_";
                        } else {
                            decoded << '<' << str[i];
                        }
                    } else if (str[i] == '~' && i + 2 < str.size()) {
                        ++i;
                        if (str[i] == 'D' && str[i + 1] == '>') {
                            decoded << delimiter;
                            ++i; // Move past the '>'
                        } else if (str[i] == 'N' && str[i + 1] == '>') {
                            decoded << '\n';
                            ++i; // Move past the '>'
                        } else {
                            decoded << '<' << str[i];
                        }
                    } else {
                        decoded << '<' << str[i];
                    }
                } else {
                    decoded << str[i];
                }
            }
            return decoded.str();
        }
    };

}
#endif //HAPPYML_TEXT_ENCODER_DECODER_HPP
