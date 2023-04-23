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
#include "../types/materialized_tensors.hpp"
#include "column_metadata.hpp"
#include "text_file_encoder_decoder.hpp"


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
        explicit DelimitedTextFileReader(const string &path, char delimiter = ',', bool skip_header = false) :
                lineReader(path,
                           skip_header) {
            this->delimiter_ = delimiter;
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
            string current_word;
            bool in_quotes = false;

            while (lineReader.hasNext()) {
                string line = lineReader.nextLine();
                for (size_t i = 0; i < line.size(); ++i) {
                    char c = line[i];
                    if (c == delimiter_ && !in_quotes) {
                        result.push_back(TextFileEncoderDecoder::decodeString(current_word, delimiter_));
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
        char delimiter_;
    };


    class BinaryDatasetReader {
    public:
        explicit BinaryDatasetReader(const string &path) : path_(path) {
            binaryFile_.open(path, ios::binary | ios::in);
            readHeader();
        }

        ~BinaryDatasetReader() {
            close();
        }

        void close() {
            if (binaryFile_.is_open()) {
                binaryFile_.close();
            }
        }

        [[nodiscard]] size_t rowCount() const {
            return number_of_rows_;
        }

        pair<vector<shared_ptr<BaseTensor>>, vector<shared_ptr<BaseTensor>>> readRow(size_t index) {
            // advance to the beginning of the row, which is header_size_ + index * row_size_
            std::streamoff offset = static_cast<std::streamoff>(index) * static_cast<std::streamoff>(row_size_);
            binaryFile_.seekg(header_size_ + offset, ios::beg);

            vector<shared_ptr<BaseTensor>> given_tensors;
            for (const auto &metadata: given_metadata_) {
                shared_ptr<BaseTensor> next_tensor = loadTensor(metadata);
                given_tensors.push_back(next_tensor);
            }
            vector<shared_ptr<BaseTensor>> expected_tensors;
            for (const auto &metadata: expected_metadata_) {
                shared_ptr<BaseTensor> next_tensor = loadTensor(metadata);
                expected_tensors.push_back(next_tensor);
            }
            return {given_tensors, expected_tensors};
        }


        char getExpectedTensorPurpose(size_t index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index]->purpose;
        }

        char getGivenTensorPurpose(size_t index) {
            if (index >= given_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return given_metadata_[index]->purpose;
        }

        vector<size_t> getExpectedTensorDims(size_t index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return {expected_metadata_[index]->rows, expected_metadata_[index]->columns, expected_metadata_[index]->channels};
        }

        vector<size_t> getGivenTensorDims(size_t index) {
            if (index >= given_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return {given_metadata_[index]->rows, given_metadata_[index]->columns, given_metadata_[index]->channels};
        }

    private:
        ifstream binaryFile_;
        size_t row_size_{};
        streampos header_size_;
        vector<shared_ptr<BinaryColumnMetadata>> given_metadata_;
        vector<shared_ptr<BinaryColumnMetadata>> expected_metadata_;
        size_t number_of_rows_{};
        string path_;

        void readHeader() {
            row_size_ = 0;
            uint64_t number_of_given;
            binaryFile_.read(reinterpret_cast<char *>(&number_of_given), sizeof(uint64_t));
            for (size_t i = 0; i < number_of_given; i++) {
                shared_ptr<BinaryColumnMetadata> metadata = readColumnMetadata();
                row_size_ += metadata->rows * metadata->columns * metadata->channels * sizeof(float);
                given_metadata_.emplace_back(metadata);
            }
            uint64_t number_of_expected;
            binaryFile_.read(reinterpret_cast<char *>(&number_of_expected), sizeof(uint64_t));
            for (size_t i = 0; i < number_of_expected; i++) {
                shared_ptr<BinaryColumnMetadata> metadata = readColumnMetadata();
                row_size_ += metadata->rows * metadata->columns * metadata->channels * sizeof(float);
                expected_metadata_.emplace_back(metadata);
            }
            header_size_ = binaryFile_.tellg();
            number_of_rows_ = static_cast<size_t>(filesystem::file_size(path_) - header_size_) / row_size_;
        }

        shared_ptr<BinaryColumnMetadata> readColumnMetadata() {
            shared_ptr<BinaryColumnMetadata> metadata = make_shared<BinaryColumnMetadata>();
            binaryFile_.read(reinterpret_cast<char *>(&metadata->purpose), sizeof(char));

            binaryFile_.read(reinterpret_cast<char *>(&metadata->is_standardized), sizeof(bool));
            uint32_t portableMean;
            binaryFile_.read(reinterpret_cast<char *>(&portableMean), sizeof(uint32_t));
            portableMean = portableBytes(portableMean);
            metadata->mean = *(float *) &portableMean;
            uint32_t portableStandardDeviation;
            binaryFile_.read(reinterpret_cast<char *>(&portableStandardDeviation), sizeof(uint32_t));
            metadata->standard_deviation = portableFloat(portableStandardDeviation);

            binaryFile_.read(reinterpret_cast<char *>(&metadata->is_normalized), sizeof(bool));
            uint32_t portableMinValue;
            binaryFile_.read(reinterpret_cast<char *>(&portableMinValue), sizeof(uint32_t));
            metadata->min_value = portableFloat(portableMinValue);
            uint32_t portableMaxValue;
            binaryFile_.read(reinterpret_cast<char *>(&portableMaxValue), sizeof(uint32_t));
            metadata->max_value = portableFloat(portableMaxValue);

            size_t portableRows;
            binaryFile_.read(reinterpret_cast<char *>(&portableRows), sizeof(uint64_t));
            metadata->rows = portableBytes(portableRows);
            size_t portableColumns;
            binaryFile_.read(reinterpret_cast<char *>(&portableColumns), sizeof(uint64_t));
            metadata->columns = portableBytes(portableColumns);
            size_t portableChannels;
            binaryFile_.read(reinterpret_cast<char *>(&portableChannels), sizeof(uint64_t));
            metadata->channels = portableBytes(portableChannels);
            return metadata;
        }

        [[nodiscard]] shared_ptr<BaseTensor> loadTensor(const shared_ptr<BinaryColumnMetadata> &metadata) {
            shared_ptr<BaseTensor> next_tensor;
            if (metadata->purpose == 'I') {
                // Pixel tensors are 8-bit unsigned integers in memory.
                next_tensor = make_shared<PixelTensor>(binaryFile_, metadata->rows, metadata->columns, metadata->channels);
            } else if (metadata->purpose == 'L') {
                // Labels are one-hot-encoded, so we can use quarter tensors.
                // Quarter tensors are 8-bit floats in memory. This is fine since labels are 0s and 1s.
                // A bias of 4 can represent 0s and 1s.
                next_tensor = make_shared<QuarterTensor>(binaryFile_, 4, metadata->rows, metadata->columns, metadata->channels);
            } else {
                next_tensor = make_shared<FullTensor>(binaryFile_, metadata->rows, metadata->columns, metadata->channels);
            }
            return next_tensor;
        }
    };
}
#endif //HAPPYML_FILE_READER_HPP
