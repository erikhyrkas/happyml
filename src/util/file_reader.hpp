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
#include <filesystem>
#include <variant>
#include "../ml/byte_pair_encoder.hpp"
#include "column_metadata.hpp"
#include "text_encoder_decoder.hpp"
#include "../training_data/data_encoder.hpp"


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

        bool is_open() {
            return stream.is_open();
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

        bool is_open() {
            return lineReader.is_open();
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
                        result.push_back(decode(current_word));
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
            result.push_back(decode(current_word));
            return result;
        }

        string decode(string &current_word) const {
            auto stripped_column = strip(current_word);
            auto stripped_column_length = stripped_column.length();
            if (stripped_column_length > 0 && stripped_column[0] == '"' &&
                stripped_column[stripped_column_length - 1] == '"') {
                stripped_column = stripped_column.substr(1, stripped_column_length - 2);
            }
            string_replace_all(stripped_column, "\"\"", "\"");
            stripped_column = TextEncoderDecoder::decodeString(stripped_column, delimiter_);
            return stripped_column;
        }

    private:
        TextLinePathReader lineReader;
        char delimiter_;
    };


    class BinaryDatasetReader {
    public:
        explicit BinaryDatasetReader(const string &path) : BinaryDatasetReader(path, {}, {}) {
        }

        explicit BinaryDatasetReader(const string &path,
                                     const std::vector<shared_ptr<BinaryColumnMetadata>> &renormalize_given_metadata,
                                     const std::vector<shared_ptr<BinaryColumnMetadata>> &renormalize_expected_metadata)
                : path_(path),
                  renormalize_given_metadata_(renormalize_given_metadata),
                  renormalize_expected_metadata_(renormalize_expected_metadata) {
            binaryFile_.open(path, ios::binary | ios::in);
            if (!binaryFile_.is_open()) {
                throw runtime_error("Could not open file " + path);
            }
            readHeader();
            if (!renormalize_given_metadata_.empty() && renormalize_given_metadata_.size() != given_metadata_.size()) {
                close();
                throw runtime_error("Incompatible given metadata for renormalization");
            }
            if (!renormalize_expected_metadata_.empty() && renormalize_expected_metadata_.size() != expected_metadata_.size()) {
                close();
                throw runtime_error("Incompatible expected metadata for renormalization");
            }

        }

        ~BinaryDatasetReader() {
            close();
        }

        bool is_open() {
            return binaryFile_.is_open();
        }

        void close() {
            if (binaryFile_.is_open()) {
                binaryFile_.close();
            }
        }

        [[nodiscard]] size_t rowCount() const {
            return number_of_rows_;
        }

        pair<vector<shared_ptr<BaseTensor>>, vector<shared_ptr<BaseTensor>>> readRow(size_t
                                                                                     index) {
            if (index >= number_of_rows_) {
                throw runtime_error("Index out of bounds");
            }
            // advance to the beginning of the row, which is header_size_ + index * row_size_
            std::streamoff offset = static_cast<std::streamoff>(index) * static_cast<std::streamoff>(row_size_);
            binaryFile_.seekg(header_size_ + offset, ios::beg);

            vector<shared_ptr<BaseTensor>> given_tensors;
            size_t given_offset = 0;
            for (const auto &metadata: given_metadata_) {
                shared_ptr<BaseTensor> next_tensor = loadTensor(metadata);
                if (!renormalize_given_metadata_.empty()) {
                    next_tensor = renormalizeAndStandardize(next_tensor, metadata->is_normalized, metadata->is_standardized,
                                                            metadata->min_value, metadata->max_value,
                                                            metadata->mean, metadata->standard_deviation,
                                                            renormalize_given_metadata_[given_offset]->is_normalized, renormalize_given_metadata_[given_offset]->is_standardized,
                                                            renormalize_given_metadata_[given_offset]->min_value, renormalize_given_metadata_[given_offset]->max_value,
                                                            renormalize_given_metadata_[given_offset]->mean, renormalize_given_metadata_[given_offset]->standard_deviation);
                }
                given_tensors.push_back(next_tensor);
            }
            vector<shared_ptr<BaseTensor>> expected_tensors;
            for (const auto &metadata: expected_metadata_) {
                shared_ptr<BaseTensor> next_tensor = loadTensor(metadata);
                if (!renormalize_expected_metadata_.empty()) {
                    next_tensor = renormalizeAndStandardize(next_tensor, metadata->is_normalized, metadata->is_standardized,
                                                            metadata->min_value, metadata->max_value,
                                                            metadata->mean, metadata->standard_deviation,
                                                            renormalize_expected_metadata_[given_offset]->is_normalized, renormalize_expected_metadata_[given_offset]->is_standardized,
                                                            renormalize_expected_metadata_[given_offset]->min_value, renormalize_expected_metadata_[given_offset]->max_value,
                                                            renormalize_expected_metadata_[given_offset]->mean, renormalize_expected_metadata_[given_offset]->standard_deviation);
                }
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

        size_t get_given_column_count() {
            return given_metadata_.size();
        }

        size_t get_expected_column_count() {
            return expected_metadata_.size();
        }

        bool is_standardized(size_t index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index]->is_standardized;
        }

        bool is_normalized(size_t index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index]->is_normalized;
        }

        shared_ptr<BinaryColumnMetadata> get_given_metadata(size_t index) {
            if (index >= given_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return given_metadata_[index];
        }

        shared_ptr<BinaryColumnMetadata> get_expected_metadata(size_t index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index];
        }


        std::vector<string> getExpectedTensorOrderedLabels(int index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index]->ordered_labels;
        }

        std::vector<string> getGivenTensorOrderedLabels(int index) {
            if (index >= given_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return given_metadata_[index]->ordered_labels;
        }

        string get_given_name(int index) {
            if (index >= given_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return given_metadata_[index]->name;
        }

        string get_expected_name(int index) {
            if (index >= expected_metadata_.size()) {
                throw runtime_error("Index out of bounds");
            }
            return expected_metadata_[index]->name;
        }

        std::vector<string> get_given_names() {
            vector<string> names;
            names.reserve(given_metadata_.size());
            for (const auto &metadata: given_metadata_) {
                names.push_back(metadata->name);
            }
            return names;
        }

        std::vector<string> get_expected_names() {
            vector<string> names;
            names.reserve(expected_metadata_.size());
            for (const auto &metadata: expected_metadata_) {
                names.push_back(metadata->name);
            }
            return names;
        }

        std::vector<shared_ptr<BinaryColumnMetadata>> get_given_metadata() {
            return given_metadata_;
        }

        std::vector<shared_ptr<BinaryColumnMetadata>> get_expected_metadata() {
            return expected_metadata_;
        }

    private:
        ifstream binaryFile_;
        size_t row_size_{};
        streampos header_size_;
        std::vector<shared_ptr<BinaryColumnMetadata>> given_metadata_;
        std::vector<shared_ptr<BinaryColumnMetadata>> expected_metadata_;
        size_t number_of_rows_{};
        string path_;
        std::vector<shared_ptr<BinaryColumnMetadata>> renormalize_given_metadata_;
        std::vector<shared_ptr<BinaryColumnMetadata>> renormalize_expected_metadata_;


        void readHeader() {
            row_size_ = 0;
            uint64_t number_of_given;
            if (!binaryFile_.read(reinterpret_cast<char *>(&number_of_given), sizeof(uint64_t))) {
                throw runtime_error("Could not read number of given tensors");
            }
            for (size_t i = 0; i < number_of_given; i++) {
                shared_ptr<BinaryColumnMetadata> metadata = readColumnMetadata();
                row_size_ += metadata->rows * metadata->columns * metadata->channels * sizeof(float);
                given_metadata_.emplace_back(metadata);
            }
            uint64_t number_of_expected;
            if (!binaryFile_.read(reinterpret_cast<char *>(&number_of_expected), sizeof(uint64_t))) {
                throw runtime_error("Could not read number of expected tensors");
            }
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

            uint64_t portable_source_column_count;
            binaryFile_.read(reinterpret_cast<char *>(&portable_source_column_count), sizeof(uint64_t));
            metadata->source_column_count = portableBytes(portable_source_column_count);

            uint64_t portableRows;
            binaryFile_.read(reinterpret_cast<char *>(&portableRows), sizeof(uint64_t));
            metadata->rows = portableBytes(portableRows);
            uint64_t portableColumns;
            binaryFile_.read(reinterpret_cast<char *>(&portableColumns), sizeof(uint64_t));
            metadata->columns = portableBytes(portableColumns);
            uint64_t portableChannels;
            binaryFile_.read(reinterpret_cast<char *>(&portableChannels), sizeof(uint64_t));
            metadata->channels = portableBytes(portableChannels);

            uint64_t portable_label_count;
            binaryFile_.read(reinterpret_cast<char *>(&portable_label_count), sizeof(uint64_t));
            size_t label_count = portableBytes(portable_label_count);
            for (size_t next_label = 0; next_label < label_count; next_label++) {
                uint64_t label_length;
                binaryFile_.read(reinterpret_cast<char *>(&label_length), sizeof(uint64_t));
                auto portable_label_length = portableBytes(label_length);
                auto label_length_streamsize = static_cast<streamsize>(portable_label_length);
                vector<char> char_array(label_length_streamsize, 0);
                binaryFile_.read(char_array.data(), label_length_streamsize);
                string label(char_array.data(), label_length_streamsize);
                metadata->ordered_labels.push_back(label);
            }

            uint64_t column_name_length;
            binaryFile_.read(reinterpret_cast<char *>(&column_name_length), sizeof(uint64_t));
            auto portable_column_name_length = portableBytes(column_name_length);
            auto column_name_length_streamsize = static_cast<streamsize>(portable_column_name_length);
            std::vector<char> column_name_char_array(column_name_length_streamsize, 0);
            binaryFile_.read(column_name_char_array.data(), column_name_length_streamsize);
            string column_name(column_name_char_array.begin(), column_name_char_array.end());
            metadata->name = column_name;

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

    vector<vector<string>> read_config(const string &directory, const string &file_name) {
        string modelProperties = directory + "/" + file_name;
        auto reader = make_unique<DelimitedTextFileReader>(modelProperties, ':');
        vector<vector<string>> metadata;
        while (reader->hasNext()) {
            metadata.push_back(reader->nextRecord());
        }
        return metadata;
    }
}
#endif //HAPPYML_FILE_READER_HPP
