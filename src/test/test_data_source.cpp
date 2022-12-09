//
// Created by Erik Hyrkas on 11/6/2022.
//
#include <iostream>
#include "../util/unit_test.hpp"
#include "../training_data/training_dataset.hpp"

using namespace microml;
using namespace std;

void test_addition_source() {
    TestAdditionGeneratedDataSource testAdditionGeneratedDataSource(10);
    std::shared_ptr<TrainingPair> next_record;
    do {
        next_record = testAdditionGeneratedDataSource.next_record();
        if (next_record) {
            std::cout << "GIVEN: " << std::endl;
            for (const auto &given: next_record->getGiven()) {
                given->print();
            }
            std::cout << "EXPECTED: " << std::endl;
            for (size_t i = 0; i < next_record->getExpectedSize(); i++) {
                for (const auto &expected: next_record->getExpected()) {
                    expected->print();
                }
            }

        }
    } while (next_record);

//    ASSERT_TRUE(test_one_quarter(NAN, 4));
}

int main() {
    try {
        test_addition_source();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
