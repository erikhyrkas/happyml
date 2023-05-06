//
// Created by Erik Hyrkas on 4/13/2023.
//

#include "../util/timers.hpp"
#include "../lang/code_block_statement.hpp"
#include "../util/unit_test.hpp"
#include "../lang/create_dataset_statement.hpp"

using namespace std;
using namespace happyml;

void test_multi_input_multi_output_create() {
    auto executionContext = make_shared<ExecutionContext>();
    vector<shared_ptr<ColumnGroup>> columnGroups;
    columnGroups.emplace_back(make_shared<ColumnGroup>(1, 0, 1, "given", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(2, 1, 1, "given", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(3, 2, 1, "given", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(4, 3, 1, "expected", "number", 1, 1, 1));
    CreateDatasetStatement test("test2", "file://../test_data/unit_test_1.csv", true, columnGroups);
    auto result = test.execute(executionContext);
    ASSERT_TRUE(result->isSuccessful());
}


void test_multi_input_multi_output_create_2() {
// Simple, wrapped syntax:
//     create dataset test \
//     with header \
//     with given number at 0 \
//     with expected number at 1 \
//     with given number at 2 \
//     with given number at 3 \
//     using file://../test_data/unit_test_1.csv
// or
//     create dataset test with header with given number(1) at 0 with expected number(1) at 1 with given number(1) at 2 with given number(1) at 3 using file://../test_data/unit_test_1.csv
//     create dataset test with header with given number(1, 1) at 0 with expected number(1, 1) at 1 with given number(1, 1) at 2 with given number(1, 1) at 3 using file://../test_data/unit_test_1.csv
//     create dataset test with header with given number(1, 1, 1) at 0 with expected number(1, 1, 1) at 1 with given number(1, 1, 1) at 2 with given number(1, 1, 1) at 3 using file://../test_data/unit_test_1.csv

    auto executionContext = make_shared<ExecutionContext>();
    vector<shared_ptr<ColumnGroup>> columnGroups;
    columnGroups.emplace_back(make_shared<ColumnGroup>(1, 0, 1, "given", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(2, 1, 1, "expected", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(3, 2, 1, "given", "number", 1, 1, 1));
    columnGroups.emplace_back(make_shared<ColumnGroup>(4, 3, 1, "given", "number", 1, 1, 1));
    CreateDatasetStatement test("test", "file://../test_data/unit_test_1.csv", true, columnGroups);
    auto result = test.execute(executionContext);
    ASSERT_TRUE(result->isSuccessful());
    string base_path = DEFAULT_HAPPYML_DATASETS_PATH;
    string result_path = base_path + "test/dataset.bin";
    BinaryDatasetReader reader(result_path);
    auto given_column_count = reader.get_given_column_count();
    ASSERT_TRUE(given_column_count == 3);
    auto expected_column_count = reader.get_expected_column_count();
    ASSERT_TRUE(expected_column_count == 1);
    auto row_count = reader.rowCount();
    ASSERT_TRUE(row_count == 2);
    for (size_t i = 0; i < row_count; i++) {
        auto given_expected = reader.readRow(i);
        auto given = given_expected.first;
        auto expected = given_expected.second;
        ASSERT_TRUE(given.size() == 3);
        ASSERT_TRUE(expected.size() == 1);
//        cout << "GIVEN: " << given[0]->getValue(0, 0, 0) << ", " << given[1]->getValue(0, 0, 0) << ", " << given[2]->getValue(0, 0, 0) << endl;
//        cout << "EXPECTED: " << expected[0]->getValue(0, 0, 0) << endl;
//        auto original_0 = unstandardize_and_denormalize(given[0], reader.get_given_metadata(0));
//        auto original_1 = unstandardize_and_denormalize(given[1], reader.get_given_metadata(1));
//        auto original_2 = unstandardize_and_denormalize(given[2], reader.get_given_metadata(2));
//        auto original_3 = unstandardize_and_denormalize(expected[0], reader.get_expected_metadata(0));
//        cout << "ORIGINAL GIVEN: " << original_0->getValue(0, 0, 0) << ", " << original_1->getValue(0, 0, 0) << ", " << original_2->getValue(0, 0, 0) << endl;
//        cout << "ORIGINAL EXPECTED: " << original_3->getValue(0, 0, 0) << endl;
    }
    // First row
    //ORIGINAL GIVEN: 0, 25, 8
    //ORIGINAL EXPECTED: 1
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(0).first[0], reader.get_given_metadata(0))->getValue(0, 0, 0), 0.0f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(0).first[1], reader.get_given_metadata(1))->getValue(0, 0, 0), 25.0f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(0).first[2], reader.get_given_metadata(2))->getValue(0, 0, 0), 8.0f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(0).second[0], reader.get_expected_metadata(0))->getValue(0, 0, 0), 1.0f));
    // Second row
    //ORIGINAL GIVEN: 1.1, 0, 0.4
    //ORIGINAL EXPECTED: 21
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(1).first[0], reader.get_given_metadata(0))->getValue(0, 0, 0), 1.1f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(1).first[1], reader.get_given_metadata(1))->getValue(0, 0, 0), 0.0f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(1).first[2], reader.get_given_metadata(2))->getValue(0, 0, 0), 0.4f));
    ASSERT_TRUE(roughlyEqual(unstandardize_and_denormalize(reader.readRow(1).second[0], reader.get_expected_metadata(0))->getValue(0, 0, 0), 21.0f));
}

void test_sing_input_multi_output_create() {

}

void test_multi_input_single_output_create() {

}

void test_single_input_single_output_create() {

}


int main() {
    try {
        EvenMoreSimpleTimer timer;

        test_multi_input_multi_output_create();
        timer.printMilliseconds();

        test_multi_input_multi_output_create_2();
        timer.printMilliseconds();

        test_sing_input_multi_output_create();
        timer.printMilliseconds();

        test_multi_input_single_output_create();
        timer.printMilliseconds();

        test_single_input_single_output_create();
        timer.printMilliseconds();

    } catch (const exception &e) {
        cout << e.what() << endl;
    }
    return 0;
}