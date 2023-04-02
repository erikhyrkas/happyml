//
// Created by Erik Hyrkas on 11/27/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FILE_READER_HPP
#define HAPPYML_FILE_READER_HPP

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <filesystem>
#include <variant>
#include "../ml/byte_pair_encoder.hpp"

using namespace std;

namespace happyml {
    class TextLinePathReader {
    public:
        explicit TextLinePathReader(const string &path, bool skip_header = false)
                : skip_header(skip_header) {
            if (filesystem::is_directory(path)) {
                for (const auto &entry: filesystem::directory_iterator(path)) {
                    if (entry.is_regular_file()) {
                        filenames.push_back(entry.path().string());
                    }
                }
            } else {
                filenames.push_back(path);
            }
            current_file = 0;
            if (filenames.empty()) {
                has_next = false;
            } else {
                stream.open(filenames[current_file]);
                has_next = stream.is_open();
                if (has_next) {
                    if (skip_header) {
                        skipHeader();
                    }
                    nextLine(); // Buffer one line and update has_next
                }
            }
        }

        ~TextLinePathReader() {
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

        string nextLine() {
            string result = next_line;
            if (has_next) {
                if (!stream.is_open()) {
                    has_next = false;
                    next_line = "";
                } else {
                    string line;
                    if (getline(stream, line)) {
                        next_line = line;
                    } else {
                        stream.close();
                        current_file++;
                        if (current_file < filenames.size()) {
                            stream.open(filenames[current_file]);
                            if (skip_header) {
                                skipHeader();
                            }
                            if (getline(stream, line)) {
                                next_line = line;
                            } else {
                                has_next = false;
                                next_line = "";
                            }
                        } else {
                            has_next = false;
                            next_line = "";
                        }
                    }
                }
            }
            return result;
        }

    private:
        void skipHeader() {
            string line;
            if (!stream.is_open() || !getline(stream, line)) {
                has_next = false;
            }
        }

        vector<string> filenames;
        size_t current_file;
        ifstream stream;
        bool has_next;
        string next_line;
        bool skip_header;
    };

    class DelimitedTextFileReader {
    public:
        explicit DelimitedTextFileReader(const string &path, char delimiter=',', bool skip_header = false) : lineReader(path,
                                                                                                           skip_header) {
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
            string current_word = "";
            bool in_quotes = false;

            while (lineReader.hasNext()) {
                string line = lineReader.nextLine();
                for (size_t i = 0; i < line.size(); ++i) {
                    char c = line[i];
                    if (c == delimiter && !in_quotes) {
                        result.push_back(current_word);
                        current_word = "";
                    } else if (c == '"') {
                        if (!in_quotes) {
                            in_quotes = true;
                        } else {
                            if (i < line.size() - 1 && line[i + 1] == '"') {
                                current_word += c;
                                ++i; // skip the next quote
                            } else {
                                in_quotes = false;
                            }
                        }
                    } else {
                        current_word += c;
                    }
                }
                if (!in_quotes) {
                    break;
                } else {
                    current_word += '\n';
                }
            }
            if (in_quotes) {
                throw runtime_error("Malformed input: unclosed quote");
            }
            result.push_back(current_word);
            return result;
        }

    private:
        TextLinePathReader lineReader;
        char delimiter;
    };

    class BinaryDatasetReader {
    public:
        BinaryDatasetReader(const string &inputPath, shared_ptr<BytePairEncoderModel> bpe) :
                bpeModel_(std::move(bpe)) {
            binaryFile_.open(inputPath, ios::binary | ios::in);
            readHeader();
        }

        ~BinaryDatasetReader() {
            binaryFile_.close();
        }

        [[nodiscard]] bool hasNext() const {
            return currentPosition_ < recordCount_;
        }

        // Method to get the total number of records in the file
        [[nodiscard]] size_t getRecordCount() const {
            return recordCount_;
        }

        // Method to fetch a specific record by its index
        vector<variant<double, string>> fetchRecord(size_t index) {
            if (index >= recordCount_) {
                throw out_of_range("Requested index is out of range.");
            }
            // Move to the requested index
            binaryFile_.seekg(static_cast<long long>(headerSize_ + index * recordSize_), ios::beg);

            // Read the requested record
            vector<variant<double, string>> record;
            record.reserve(columnTypes_.size());

            for (size_t i = 0; i < columnTypes_.size(); ++i) {
                if (columnTypes_[i] == 'N') {
                    double number;
                    binaryFile_.read(reinterpret_cast<char *>(&number), sizeof(double));
                    record.emplace_back(number);
                } else {
                    size_t encodedSize = columnWidths_[i] / sizeof(char16_t);
                    vector<char16_t> encoded(encodedSize);
                    binaryFile_.read(reinterpret_cast<char *>(encoded.data()), static_cast<streamsize>(columnWidths_[i]));
                    u16string encodedStr(encoded.begin(), encoded.end());
                    auto first_null = encodedStr.find(u'\0');
                    if( first_null != string::npos) {
                        encodedStr.erase(first_null, string::npos);  // Remove padding
                    }
                    string text = bpeModel_->decode(encodedStr);
                    record.emplace_back(text);
                }
            }

            currentPosition_ = index + 1;
            return record;
        }

        vector<variant<double, string>> nextMixedRecord() {
            return fetchRecord(currentPosition_);
        }

        vector<double> nextDoubleRecord() {
            vector<variant<double, string>> mixedRecord = fetchRecord(currentPosition_);
            vector<double> record;
            record.reserve(mixedRecord.size());

            for (const auto &value : mixedRecord) {
                if (holds_alternative<double>(value)) {
                    record.push_back(get<double>(value));
                } else {
                    throw runtime_error("This record contains a non-numeric value.");
                }
            }

            return record;
        }

        // Method that checks if all columns are of the numeric type
        [[nodiscard]] bool areAllColumnsNumbers() const {
            return all_of(columnTypes_.begin(), columnTypes_.end(), [](char colType) {
                return colType == 'N';
            });
        }

        vector<string> nextRecord() {
            vector<variant<double, string>> mixedRecord = fetchRecord(currentPosition_);
            vector<string> record;
            record.reserve(mixedRecord.size());

            for (const auto &value : mixedRecord) {
                if (holds_alternative<double>(value)) {
                    record.push_back(to_string(get<double>(value)));
                } else {
                    record.emplace_back(get<string>(value));
                }
            }

            return record;
        }

    private:
        ifstream binaryFile_;
        shared_ptr<BytePairEncoderModel> bpeModel_;
        vector<char> columnTypes_;
        vector<size_t> columnWidths_;
        size_t recordCount_{};
        size_t currentPosition_ = 0;
        size_t headerSize_ = 0;
        size_t recordSize_ = 0;

        void readHeader() {
            // Read column types
            uint16_t columnTypesSize;
            binaryFile_.read(reinterpret_cast<char *>(&columnTypesSize), sizeof(uint16_t));
            columnTypes_.resize(columnTypesSize);
            binaryFile_.read(reinterpret_cast<char *>(columnTypes_.data()), static_cast<streamsize>(columnTypesSize * sizeof(char)));

            // Read column widths
            size_t columnWidthsSize = columnTypes_.size();
            columnWidths_.resize(columnWidthsSize);
            binaryFile_.read(reinterpret_cast<char *>(columnWidths_.data()), static_cast<streamsize>(columnWidthsSize * sizeof(size_t)));
            recordSize_ = accumulate(columnWidths_.begin(), columnWidths_.end(), size_t{0});

            // Read record count
            binaryFile_.read(reinterpret_cast<char *>(&recordCount_), sizeof(size_t));

            // Read BPE model name
            uint16_t bpeModelNameLength;

            binaryFile_.read(reinterpret_cast<char *>(&bpeModelNameLength), sizeof(uint16_t));
            string bpeModelName(bpeModelNameLength, '\0');
            binaryFile_.read(&bpeModelName[0], static_cast<streamsize>(bpeModelNameLength * sizeof(char)));

            // Verify BPE model name matches
            if (bpeModel_->getName() != bpeModelName) {
                throw runtime_error("BPE model name in binary file does not match provided BPE model.");
            }
            headerSize_ = binaryFile_.tellg();
        }
    };
}
#endif //HAPPYML_FILE_READER_HPP
