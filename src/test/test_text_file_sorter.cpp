//
// Created by Erik Hyrkas on 4/22/2023.
//

#include <iostream>
#include "../util/unit_test.hpp"
#include "../util/text_file_sorter.hpp"
#include "../util/timers.hpp"

using namespace std;
using namespace happyml;

void test_file_sort1() {
    if (!FileSorter::sort("../test_data/duplicate_test.txt", "../test_data/duplicate_test_sorted_has_header.txt", true, 5, true)) {
        throw exception("Missing file.");
    }
    // count lines in duplicate_test_sorted_no_header.txt
    ifstream file("../test_data/duplicate_test_sorted_has_header.txt");
    string line;
    int line_count = 0;
    while (getline(file, line)) {
        ++line_count;
    }
    file.close();
    ASSERT_TRUE(line_count == 3);
    filesystem::remove("../test_data/duplicate_test_sorted_has_header.txt");
    PASS_TEST();
}

void test_file_sort2() {
    if (!FileSorter::sort("../test_data/duplicate_test.txt", "../test_data/duplicate_test_sorted_whole_file.txt", false,1, true)) {
        throw exception("Missing file.");
    }

    ifstream file("../test_data/duplicate_test_sorted_whole_file.txt");
    string line;
    int line_count = 0;
    while (getline(file, line)) {
        ++line_count;
    }
    file.close();
    ASSERT_TRUE(line_count == 3);
    filesystem::remove("../test_data/duplicate_test_sorted_whole_file.txt");
    PASS_TEST();
}
void test_file_sort3() {
    if (!FileSorter::sort("../test_data/duplicate_test.txt", "../test_data/duplicate_test_sorted_keep_duplicates.txt", true, 5, false)) {
        throw exception("Missing file.");
    }
    // count lines in duplicate_test_sorted_no_header.txt
    ifstream file("../test_data/duplicate_test_sorted_keep_duplicates.txt");
    string line;
    int line_count = 0;
    while (getline(file, line)) {
        ++line_count;
    }
    file.close();
    ASSERT_TRUE(line_count == 16);
    filesystem::remove("../test_data/duplicate_test_sorted_keep_duplicates.txt");
    PASS_TEST();
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_file_sort1();
        timer.printMilliseconds();
        test_file_sort2();
        timer.printMilliseconds();
        test_file_sort3();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}