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
#include "file_reader.hpp"
#include "../ml/byte_pair_encoder.hpp"

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
        DelimitedTextFileWriter(const string &path, char delimiter) : line_writer_(path), delimiter_(delimiter) {
        }

        ~DelimitedTextFileWriter() {
            close();
        }

        void close() {
            line_writer_.close();
        }

        void writeRecord(const vector<string> &record) {
            string currentDelimiter;
            stringstream combinedRecord;
            for (const string &column: record) {
                combinedRecord << currentDelimiter << column;
                currentDelimiter = delimiter_;
            }
            line_writer_.writeLine(combinedRecord.str());
        }

    private:
        TextLineFileWriter line_writer_;
        char delimiter_;
    };

    class BinaryDatasetWriter {
    public:
        BinaryDatasetWriter(const string &path, shared_ptr<BytePairEncoderModel> bpe) : bpeModel_(std::move(bpe)) {
            binaryFile_.open(path, ios::binary | ios::out);
        }

        ~BinaryDatasetWriter() {
            close();
        }

        void close() {
            if (binaryFile_.is_open()) {
                binaryFile_.close();
            }
        }

        void calculateAndWriteHeader(vector<vector<std::variant<double, string>>> &records) {
            // Detect column types and widths.
            columnTypes_.clear();
            columnWidths_.clear();
            size_t recordCount = records.size();
            detectColumnTypes(records[0], columnTypes_);
            for (const auto &record: records) {
                updateColumnWidths(record, columnTypes_, columnWidths_);
            }

            // Write the header
            writeHeader(columnTypes_, columnWidths_, recordCount, bpeModel_->getName());
        }

        void writeRecord(const vector<std::variant<double, string>> &record) {
            writeRecordInternal(record, columnTypes_, columnWidths_);
        }

        static void detectColumnTypes(const vector<std::variant<double, string>> &record, vector<char> &columnTypes) {
            for (const auto &cell: record) {
                columnTypes.push_back(holds_alternative<double>(cell) ? 'N' : 'T');
            }
        }

        static bool isNumber(const string &s) {
            return !s.empty() &&
                   find_if(s.begin(), s.end(), [](unsigned char c) { return !isdigit(c) && c != '.'; }) == s.end();
        }

        void updateColumnWidths(const vector<std::variant<double, string>> &record, const vector<char> &columnTypes,
                                vector<size_t> &columnWidths) {
            if (columnWidths.empty()) {
                for (size_t i = 0; i < record.size(); ++i) {
                    if (columnTypes[i] == 'N') {
                        columnWidths.push_back(sizeof(double));
                    } else {
                        u16string encoded = bpeModel_->encode(get<string>(record[i]));
                        columnWidths.push_back(encoded.size() * sizeof(char16_t));
                    }
                }
            } else {
                for (size_t i = 0; i < record.size(); ++i) {
                    if (columnTypes[i] == 'T') {
                        u16string encoded = bpeModel_->encode(get<string>(record[i]));
                        columnWidths[i] = max(columnWidths[i], encoded.size() * sizeof(char16_t));
                    }
                }
            }
        }

        void writeHeader(const vector<char> &columnTypes,
                         const vector<size_t> &columnWidths,
                         size_t recordCount,
                         const string &bpeModelName) {
            uint16_t columnTypesSize = columnTypes.size();
            binaryFile_.write(reinterpret_cast<const char *>(&columnTypesSize), sizeof(uint16_t));
            binaryFile_.write(reinterpret_cast<const char *>(columnTypes.data()),
                              static_cast<streamsize>(columnTypesSize * sizeof(char)));

            binaryFile_.write(reinterpret_cast<const char *>(&columnWidths[0]),
                              static_cast<streamsize>(columnWidths.size() * sizeof(size_t)));

            binaryFile_.write(reinterpret_cast<const char *>(&recordCount), sizeof(size_t));
            uint16_t bpeModelNameLength = bpeModelName.size();
            binaryFile_.write(reinterpret_cast<const char *>(&bpeModelNameLength), sizeof(uint16_t));
            binaryFile_.write(bpeModelName.c_str(), static_cast<streamsize>(bpeModelNameLength * sizeof(char)));
        }

        void writeRecordInternal(const vector<std::variant<double, string>> &record, const vector<char> &columnTypes,
                                 const vector<size_t> &columnWidths) {
            for (size_t i = 0; i < record.size(); ++i) {
                if (columnTypes[i] == 'N') {
                    double number = get<double>(record[i]);
                    binaryFile_.write(reinterpret_cast<const char *>(&number), sizeof(double));
                } else {
                    u16string encoded = bpeModel_->encode(get<string>(record[i]));
                    vector<char16_t> encodedResized(encoded.size());
                    copy(encoded.begin(), encoded.end(), encodedResized.begin());
                    encodedResized.resize(columnWidths[i] / sizeof(char16_t), u'\0');
                    binaryFile_.write(reinterpret_cast<const char *>(encodedResized.data()),
                                      static_cast<streamsize>(columnWidths[i]));
                }
            }
        }

        static void detectColumnTypes(const vector<string> &record, vector<char> &columnTypes) {
            for (const string &cell: record) {
                columnTypes.push_back(isNumber(cell) ? 'N' : 'T');
            }
        }

        static vector<variant<double, string>> convertStringsToVariants(
                const vector<string> &record,
                const vector<char> &columnTypes ) {
            vector<variant<double, string>> result;
            for (size_t i = 0; i < record.size(); ++i) {
                if (columnTypes[i] == 'N') {
                    result.emplace_back(stod(record[i]));
                } else {
                    result.emplace_back(record[i]);
                }
            }
            return result;
        }

        static vector<variant<double, string>> convertDoublesToVariants(
                const vector<double> &record ) {
            vector<variant<double, string>> result;
            result.reserve(record.size());
            for (const double &value: record) {
                result.emplace_back(value);
            }
            return result;
        }
    private:
        shared_ptr<BytePairEncoderModel> bpeModel_;
        ofstream binaryFile_;
        vector<char> columnTypes_;
        vector<size_t> columnWidths_;
    };

    class TextToBinaryDatasetConverter {
    public:
        explicit TextToBinaryDatasetConverter(const shared_ptr<BytePairEncoderModel> &bpe) : bpeModel_(bpe) {
        }

        void convert(const string &inputPath, const string &outputPath, char delimiter,
                     bool skip_header = false) {
            BinaryDatasetWriter writer = BinaryDatasetWriter(outputPath, bpeModel_);
            DelimitedTextFileReader textFileReader(inputPath, delimiter, skip_header);
            vector<vector<std::variant<double, string>>> records;
            vector<char> columnTypes;

            // Read input text file and convert records.
            while (textFileReader.hasNext()) {
                vector<string> textRecord = textFileReader.nextRecord();
                if (columnTypes.empty()) {
                    // Initialize column types with 'N' for each column.
                    columnTypes.assign(textRecord.size(), 'N');
                }

                // Update column types based on record values.
                for (size_t i = 0; i < textRecord.size(); ++i) {
                    if (columnTypes[i] == 'N' && !BinaryDatasetWriter::isNumber(textRecord[i])) {
                        columnTypes[i] = 'T';
                    }
                }
                records.push_back(BinaryDatasetWriter::convertStringsToVariants(textRecord, columnTypes));
            }

            textFileReader.close();

            // Calculate and write the header using BinaryDatasetWriter.
            writer.calculateAndWriteHeader(records);

            // Write out the data using BinaryDatasetWriter.
            for (const auto &record: records) {
                writer.writeRecord(record);
            }
        }

    private:
        shared_ptr<BytePairEncoderModel> bpeModel_;
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
