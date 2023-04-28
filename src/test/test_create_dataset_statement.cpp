//
// Created by Erik Hyrkas on 4/13/2023.
//

#include "../util/timers.hpp"
#include "../lang/statements.hpp"
#include "../util/unit_test.hpp"

using namespace std;
using namespace happyml;

void test_multi_input_multi_output_create() {
    auto executionContext = make_shared<ExecutionContext>();
    vector<ColumnGroup> columnGroups;
    columnGroups.emplace_back(0, "given", "number", 1, 1, 1);
    columnGroups.emplace_back(1, "given", "number", 1, 1, 1);
    columnGroups.emplace_back(2, "given", "number", 1, 1, 1);
    columnGroups.emplace_back(3, "expected", "number", 1, 1, 1);
    CreateDatasetStatement test("test", "file://../test_data/unit_test_1.csv", true, columnGroups);
    auto result = test.execute(executionContext);
    ASSERT_TRUE(result->isSuccessful());
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