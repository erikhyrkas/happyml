//
// Created by Erik Hyrkas on 11/27/2022.
//

#ifndef MICROML_FILE_READER_HPP
#define MICROML_FILE_READER_HPP

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace microml {
    class TextLineFileReader {
    public:
        explicit TextLineFileReader(const std::string &path) {
            this->filename = path;
            stream.open(filename);
            has_next = true;
            nextLine(); // buffer one line and update has_next
        }

        ~TextLineFileReader() {
            close();
        }

        void close() {
            if (stream.is_open()) {
                stream.close();
            }
        }

        [[nodiscard]] bool hasNext() const {
            return has_next;
        }

        std::string nextLine() {
            // we have read ahead one line, so we know our result.
            string result = next_line;
            // now we need to keep reading ahead if we can
            if (has_next) {
                if (!stream.is_open()) {
                    // maybe this should be an exception.
                    has_next = false;
                    next_line = "";
                } else {
                    std::string line;
                    if (!std::getline(stream, line)) {
                        // we reached the end of the file. let's release the file handle now.
                        stream.close();
                        has_next = false;
                        next_line = "";
                    } else {
                        next_line = line;
                    }
                }
            }
            return result;
        }

    private:
        std::string filename;
        std::ifstream stream;
        bool has_next;
        std::string next_line;
    };

    class DelimitedTextFileReader {
    public:
        DelimitedTextFileReader(const std::string &path, char delimiter) : lineReader(path) {
            this->delimiter = delimiter;
        }

        ~DelimitedTextFileReader() {
            close();
        }

        void close() {
            lineReader.close();
        }

        bool hasNext() {
            return lineReader.hasNext();
        }

        vector<string> nextRecord() {
            vector<string> result;
            stringstream word_stream(lineReader.nextLine());
            string word;
            while (!word_stream.eof()) {
                getline(word_stream, word, delimiter);
                result.push_back(word);
            }
            return result;
        }

    private:
        TextLineFileReader lineReader;
        char delimiter;
    };
}
#endif //MICROML_FILE_READER_HPP
