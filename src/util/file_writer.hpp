//
// Created by erikh on 12/18/2022.
//

#ifndef MICROML_FILE_WRITER_HPP
#define MICROML_FILE_WRITER_HPP

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace microml {
    class TextLineFileWriter {
    public:
        explicit TextLineFileWriter(const string &path) {
            this->filename = path;
            stream.open(filename);
        }

        ~TextLineFileWriter() {
            close();
        }

        void close() {
            if (stream.is_open()) {
                stream.close();
            }
        }

        void writeLine(const string &line) {
            if (!stream.is_open()) {
                throw exception("File is closed.");
            }
            stream << line << endl;
        }

    private:
        string filename;
        ofstream stream;
    };

    class DelimitedTextFileWriter {
    public:
        DelimitedTextFileWriter(const string &path, char delimiter) : lineWriter(path) {
            this->delimiter = delimiter;
        }

        ~DelimitedTextFileWriter() {
            close();
        }

        void close() {
            lineWriter.close();
        }

        void writeRecord(const vector<string> &record) {
            string currentDelimiter = "";
            stringstream combinedRecord;
            for( string column : record) {
                combinedRecord << currentDelimiter << column;
                currentDelimiter = delimiter;
            }
            lineWriter.writeLine(combinedRecord.str());
        }

    private:
        TextLineFileWriter lineWriter;
        char delimiter;
    };
}
#endif //MICROML_FILE_WRITER_HPP
