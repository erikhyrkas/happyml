//
// Created by Erik Hyrkas on 11/28/2022.
// Copyright 2022. Usable under MIT license.
//

#include <iostream>
#include "../util/file_reader.hpp"
#include "../util/file_writer.hpp"
#include "../util/unit_test.hpp"

using namespace std;
using namespace happyml;

void readUnitTestData() {
    auto lineReader = make_shared<TextLinePathReader>("..\\test_data\\unit_test_1.csv");
    ASSERT_TRUE(lineReader->hasNext());
    string line = lineReader->nextLine();
    ASSERT_FALSE(line.empty());
    lineReader->close();

    auto textFileReader = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_1.csv", ',');
    ASSERT_TRUE(textFileReader->hasNext());
    auto csv_record = textFileReader->nextRecord();
    ASSERT_TRUE(4 == csv_record.size());
    textFileReader->close();
}

void writeReadTest() {
    auto lineWriter = make_shared<DelimitedTextFileWriter>("..\\test_data\\unit_test_2.properties", ':');
    lineWriter->writeRecord({"name", "mymodel"});
    lineWriter->writeRecord({"size", "massive"});
    lineWriter->close();
    auto lineReader = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_2.properties", ':');
    auto firstRecord = lineReader->nextRecord();
    ASSERT_TRUE("name" == firstRecord[0]);
    ASSERT_TRUE("mymodel" == firstRecord[1]);
    auto secondRecord = lineReader->nextRecord();
    ASSERT_TRUE("size" == secondRecord[0]);
    ASSERT_TRUE("massive" == secondRecord[1]);
    remove("..\\test_data\\unit_test_2.properties");
}

void binaryTest() {
    string file_path = "..\\test_data\\unit_test_binary1.properties";
    string binary_file_path = "..\\test_data\\unit_test_binary1.bin";
    char delimiter = ':';
    auto lineWriter = make_shared<DelimitedTextFileWriter>(file_path, delimiter);
    lineWriter->writeRecord({"prop", "val"});
    lineWriter->writeRecord({"name", "mymodel"});
    lineWriter->writeRecord({"size", "tiny"});
    lineWriter->close();
    auto bpe = make_shared<BytePairEncoderModel>();
    auto converter = TextToBinaryDatasetConverter(bpe);
    converter.convert(file_path, binary_file_path, delimiter, true);
    auto reader = BinaryDatasetReader(binary_file_path, bpe);

    while(reader.hasNext()) {
        auto nextRecord = reader.nextRecord();
        cout << "binary test: [" << nextRecord[0] << "], [" << nextRecord[1] << "]" << endl;
        if( "name" == nextRecord[0]) {
            ASSERT_TRUE("mymodel" == nextRecord[1]);
        } else {
            ASSERT_TRUE("tiny" == nextRecord[1]);
        }
    }
}


void binaryTest2() {
    string file_path = "..\\test_data\\unit_test_binary2.properties";
    string binary_file_path = "..\\test_data\\unit_test_binary2.bin";
    char delimiter = ':';
    auto lineWriter = make_shared<DelimitedTextFileWriter>(file_path, delimiter);
    lineWriter->writeRecord({"0", "1.1"});
    lineWriter->writeRecord({"1", "3"});
    lineWriter->close();
    auto bpe = make_shared<BytePairEncoderModel>();
    auto converter = TextToBinaryDatasetConverter(bpe);
    converter.convert(file_path, binary_file_path, delimiter);
    auto reader = BinaryDatasetReader(binary_file_path, bpe);

    while(reader.hasNext()) {
        auto nextRecord = reader.nextDoubleRecord();
        cout << "binary test: [" << nextRecord[0] << "], [" << nextRecord[1] << "]" << endl;
        if( 0 == nextRecord[0]) {
            ASSERT_TRUE(1.1 == nextRecord[1]);
        } else {
            ASSERT_TRUE(3 == nextRecord[1]);
        }
    }
}
int main() {
    try {
        binaryTest2();
        binaryTest();
        readUnitTestData();
        writeReadTest();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}