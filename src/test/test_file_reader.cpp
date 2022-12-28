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
    auto lineReader = make_shared<TextLineFileReader>("..\\test_data\\unit_test_1.csv");
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

int main() {
    try {
//        readUnitTestData();
        writeReadTest();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}