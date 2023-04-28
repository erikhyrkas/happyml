//
// Created by Erik Hyrkas on 11/28/2022.
// Copyright 2022. Usable under MIT license.
//

#include <iostream>
#include "../util/file_reader.hpp"
#include "../util/file_writer.hpp"
#include "../util/unit_test.hpp"
#include "../util/tensor_utils.hpp"
#include "../util/dataset_utils.hpp"

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

void test_save_tensor() {
    vector<shared_ptr<BinaryColumnMetadata>> given_metadata;

    auto t1 = tensor({{{1, 2, 3}, {4, 5, 6}}});
    t1->print();

    auto metadata1 = make_shared<BinaryColumnMetadata>();
    metadata1->purpose = 'N';
    metadata1->is_normalized = false;
    metadata1->is_standardized = false;
    metadata1->rows = t1->rowCount();
    metadata1->columns = t1->columnCount();
    metadata1->channels = t1->channelCount();
    given_metadata.push_back(metadata1);


    auto t2 = tensor({{{7, 8, 9}, {10, 11, 12}}});
    t2->print();

    auto metadata2 = make_shared<BinaryColumnMetadata>();
    metadata2->purpose = 'N';
    metadata2->is_normalized = false;
    metadata2->is_standardized = false;
    metadata2->rows = t2->rowCount();
    metadata2->columns = t2->columnCount();
    metadata2->channels = t2->channelCount();
    given_metadata.push_back(metadata2);

    auto writer = make_shared<BinaryDatasetWriter>("..\\test_data\\unit_test_tensor.bin", given_metadata);
    writer->writeRow({t1, t2});
    writer->close();

    auto reader = make_shared<BinaryDatasetReader>("..\\test_data\\unit_test_tensor.bin");
    auto row = reader->readRow(0);
    reader->close();

    ASSERT_TRUE(2 == row.first.size());
    ASSERT_TRUE(row.second.empty());
    auto t1_read = row.first[0];
    auto t2_read = row.first[1];

    ASSERT_TRUE(t1->columnCount() == t1_read->columnCount());
    ASSERT_TRUE(t1->rowCount() == t1_read->rowCount());
    ASSERT_TRUE(t1->channelCount() == t1_read->channelCount());
    t1->print();
    t1_read->print();

    ASSERT_TRUE(t1->getValue(0, 0, 0) == t1_read->getValue(0, 0, 0));
    ASSERT_TRUE(t1->getValue(0, 1, 0) == t1_read->getValue(0, 1, 0));
    ASSERT_TRUE(reader->getGivenTensorPurpose(0) == 'N');

    ASSERT_TRUE(t2->columnCount() == t2_read->columnCount());
    ASSERT_TRUE(t2->rowCount() == t2_read->rowCount());
    ASSERT_TRUE(t2->channelCount() == t2_read->channelCount());
    ASSERT_TRUE(t2->getValue(0, 0, 0) == t2_read->getValue(0, 0, 0));
    ASSERT_TRUE(t2->getValue(0, 1, 0) == t2_read->getValue(0, 1, 0));
    ASSERT_TRUE(reader->getGivenTensorPurpose(1) == 'N');
    filesystem::remove("..\\test_data\\unit_test_tensor.bin");
}

void test_encode_decode_of_newlines() {
    auto textFileReader = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_2.csv", ',', true);
    auto textFileWriter = make_shared<DelimitedTextFileWriter>("..\\test_data\\unit_test_2_test.csv", ',');

    int record_count = 0;
    while(textFileReader->hasNext()) {
        record_count++;
        auto record = textFileReader->nextRecord();
        cout << "record: [" << record[0] << "], [" << record[1] << "]" << endl;
        textFileWriter->writeRecord(record);
    }

    textFileReader->close();
    textFileWriter->close();

    ASSERT_TRUE(3 == record_count);
    auto resultVerifier = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_2_test.csv", ',');
    int result_record_count = 0;
    while(resultVerifier->hasNext()) {
        result_record_count++;
        auto record = resultVerifier->nextRecord();
        cout << "record: [" << record[0] << "], [" << record[1] << "]" << endl;
    }
    resultVerifier->close();
    ASSERT_TRUE(3 == result_record_count);

    filesystem::remove("..\\test_data\\unit_test_2_test.csv");
}

void test_escaped_encode_decode_of_newlines() {
    auto textFileReader = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_3.csv", ',', true);
    auto textFileWriter = make_shared<DelimitedTextFileWriter>("..\\test_data\\unit_test_3_test.csv", ',');

    int record_count = 0;
    while(textFileReader->hasNext()) {
        record_count++;
        auto record = textFileReader->nextRecord();
        cout << "record: [" << record[0] << "], [" << record[1] << "]" << endl;
        textFileWriter->writeRecord(record);
    }

    textFileReader->close();
    textFileWriter->close();

    ASSERT_TRUE(3 == record_count);
    auto resultVerifier = make_shared<DelimitedTextFileReader>("..\\test_data\\unit_test_3_test.csv", ',');
    int result_record_count = 0;
    while(resultVerifier->hasNext()) {
        result_record_count++;
        auto record = resultVerifier->nextRecord();
        cout << "record: [" << record[0] << "], [" << record[1] << "]" << endl;
    }
    resultVerifier->close();
    ASSERT_TRUE(3 == result_record_count);

    filesystem::remove("..\\test_data\\unit_test_3_test.csv");
}

void test_convert_txt_to_csv() {
    convert_txt_to_csv("../data/data.txt", "../data/data.csv", 4000);
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_convert_txt_to_csv();
        timer.printMilliseconds();
        test_escaped_encode_decode_of_newlines();
        timer.printMilliseconds();
        test_encode_decode_of_newlines();
        timer.printMilliseconds();
        test_save_tensor();
        timer.printMilliseconds();
        readUnitTestData();
        timer.printMilliseconds();
        writeReadTest();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}