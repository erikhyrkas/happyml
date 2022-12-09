//
// Created by Erik Hyrkas on 11/28/2022.
//

#include "../util/file_reader.hpp"
#include "../util/unit_test.hpp"
#include <iostream>

using namespace std;
using namespace microml;


int main() {
    try {
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

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}