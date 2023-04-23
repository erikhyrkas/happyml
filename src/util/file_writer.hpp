//
// Created by Erik Hyrkas on 12/18/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FILE_WRITER_HPP
#define HAPPYML_FILE_WRITER_HPP

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <variant>
#include "lru_cache.h"
#include "file_reader.hpp"
#include "../ml/byte_pair_encoder.hpp"
#include "column_metadata.hpp"
#include "text_file_encoder_decoder.hpp"

using namespace std;

namespace happyml {
    class TextLineFileWriter {
    public:
        explicit TextLineFileWriter(string path) : filename_(std::move(path)) {
            stream_.open(filename_);
        }

        ~TextLineFileWriter() {
            close();
        }

        void close() {
            if (stream_.is_open()) {
                stream_.close();
            }
        }

        void writeLine(const string &line) {
            if (!stream_.is_open()) {
                throw exception("File is closed.");
            }
            stream_ << line << endl;
        }

    private:
        string filename_;
        ofstream stream_;
    };

    class DelimitedTextFileWriter {
    public:
        DelimitedTextFileWriter(const string &path, char delimiter)
                : line_writer_(path), delimiter_(delimiter) {}

        ~DelimitedTextFileWriter() {
            close();
        }

        void close() {
            line_writer_.close();
        }

        void writeRecord(const vector<string> &record) {
            string currentDelimiter;
            stringstream combinedRecord;
            for (const string &column : record) {
                combinedRecord << currentDelimiter << TextFileEncoderDecoder::encodeString(column, delimiter_);
                currentDelimiter = delimiter_;
            }
            line_writer_.writeLine(combinedRecord.str());
        }

    private:
        TextLineFileWriter line_writer_;
        char delimiter_;
    };

    // The binary dataset format is as follows:
    // 1. The header is written first:
    //      a. the number of given tensors
    //      b. each given tensor's purpose and dimensions
    //          i. tensor purpose: 'I' (image), 'T' (text), 'N' (number), 'L' (label)
    //          ii. tensor dimensions
    //      c. the number of expected tensors
    //      d. each expected tensor's purpose and dimensions
    //          i. tensor purpose: 'I' (image), 'T' (text), 'N' (number), 'L' (label)
    //          ii. tensor dimensions
    // 2. The data is written next, one row at a time:
    //      a. each given tensor's data
    //      b. each expected tensor's data
    class BinaryDatasetWriter {
    public:
        explicit BinaryDatasetWriter(const string &path,
                                     vector<shared_ptr<BinaryColumnMetadata>> given_metadata,
                                     size_t lru_cache_size = 100000) :
                BinaryDatasetWriter(path,
                                    std::move(given_metadata),
                                    {},
                                    lru_cache_size) {
        }

        explicit BinaryDatasetWriter(const string &path, vector<shared_ptr<BinaryColumnMetadata>> given_metadata,
                                     vector<shared_ptr<BinaryColumnMetadata>> expected_metadata,
                                     size_t lru_cache_size = 100000) :
                given_metadata_(std::move(given_metadata)),
                expected_metadata_(std::move(expected_metadata)) {
            if(lru_cache_size > 0) {
                lru_cache_ = make_shared<LruCache<size_t, bool>>(lru_cache_size);
            } else {
                lru_cache_ = nullptr;
            }
            binaryFile_.open(path, ios::binary | ios::out);
            writeHeader();
        }

        ~BinaryDatasetWriter() {
            close();
        }

        void close() {
            if (binaryFile_.is_open()) {
                binaryFile_.close();
            }
        }

        void writeRow(const vector<shared_ptr<BaseTensor>> &given_tensors) {
            writeRow(given_tensors, {});
        }

        void writeRow(const vector<shared_ptr<BaseTensor>> &given_tensors,
                      const vector<shared_ptr<BaseTensor>> &expected_tensors) {
            if (lru_cache_ != nullptr) {
                size_t given_hash = compute_given_hash(given_tensors);
                if (lru_cache_->contains(given_hash)) {
                    return;  // Skip writing the duplicate row
                }
                lru_cache_->insert(given_hash, true);
            }

            for (const auto &tensor: given_tensors) {
                // (8 bytes * L rows * M columns * N channels)
                tensor->save(binaryFile_, false);
            }
            for (const auto &tensor: expected_tensors) {
                tensor->save(binaryFile_, false);
            }
        }

    private:
        ofstream binaryFile_;
        vector<shared_ptr<BinaryColumnMetadata>> given_metadata_;
        vector<shared_ptr<BinaryColumnMetadata>> expected_metadata_;
        shared_ptr<LruCache < size_t, bool>> lru_cache_;

        void writeHeader() {
            uint64_t number_of_given = given_metadata_.size();
            binaryFile_.write(reinterpret_cast<const char *>(&number_of_given), sizeof(uint64_t));
            for (const auto &metadata: given_metadata_) {
                writeColumnMetadata(metadata);
            }
            uint64_t number_of_expected = expected_metadata_.size();
            binaryFile_.write(reinterpret_cast<const char *>(&number_of_expected), sizeof(uint64_t));
            for (const auto &metadata: expected_metadata_) {
                writeColumnMetadata(metadata);
            }
        }

        void writeColumnMetadata(const shared_ptr<BinaryColumnMetadata> &column_metadata) {
            binaryFile_.write(reinterpret_cast<const char *>(&column_metadata->purpose), sizeof(char));

            binaryFile_.write(reinterpret_cast<const char *>(&column_metadata->is_standardized), sizeof(bool));
            auto portableMean = portableBytes(*(uint32_t *) &column_metadata->mean);
            binaryFile_.write(reinterpret_cast<const char *>(&portableMean), sizeof(float));
            auto portalStandardDeviation = portableBytes(*(uint32_t *) &column_metadata->standard_deviation);
            binaryFile_.write(reinterpret_cast<const char *>(&portalStandardDeviation), sizeof(float));

            binaryFile_.write(reinterpret_cast<const char *>(&column_metadata->is_normalized), sizeof(bool));
            auto portableMinValue = portableBytes(*(uint32_t *) &column_metadata->min_value);
            binaryFile_.write(reinterpret_cast<const char *>(&portableMinValue), sizeof(float));
            auto portableMaxValue = portableBytes(*(uint32_t *) &column_metadata->max_value);
            binaryFile_.write(reinterpret_cast<const char *>(&portableMaxValue), sizeof(float));

            auto portableRows = portableBytes(column_metadata->rows);
            binaryFile_.write(reinterpret_cast<const char *>(&portableRows), sizeof(uint64_t));
            auto portableColumns = portableBytes(column_metadata->columns);
            binaryFile_.write(reinterpret_cast<const char *>(&portableColumns), sizeof(uint64_t));
            auto portableChannels = portableBytes(column_metadata->channels);
            binaryFile_.write(reinterpret_cast<const char *>(&portableChannels), sizeof(uint64_t));
        }

        static size_t compute_given_hash(const vector<shared_ptr<BaseTensor>> &given_tensors) {
            std::hash<string> hasher;
            size_t result = 0;
            for (const auto &next_tensor: given_tensors) {
                stringstream ss;
                next_tensor->print(ss);
                result ^= hasher(ss.str()) + 0x9e3779b9 + (result << 6) + (result >> 2);
            }
            return result;
        }
    };

    void save_config(const string &path, vector<vector<string>> &metadata) {
        if (filesystem::is_directory(path)) {
            filesystem::remove_all(path);
        }
        filesystem::create_directories(path);
        string modelProperties = path + "/configuration.happyml";
        auto writer = make_unique<DelimitedTextFileWriter>(modelProperties, ':');
        for (const auto &record: metadata) {
            writer->writeRecord(record);
        }
        writer->close();
    }
}
#endif //HAPPYML_FILE_WRITER_HPP
