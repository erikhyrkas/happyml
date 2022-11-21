//
// Created by Erik Hyrkas on 11/6/2022.
//
#include <iostream>
#include "unit_test.hpp"
#include "data_source.hpp"

void test_addition_source() {
    TestAdditionGeneratedDataSource testAdditionGeneratedDataSource(10);
    std::shared_ptr<TrainingPair> next_record;
    do {
        next_record = testAdditionGeneratedDataSource.next_record();
        if(next_record) {
            std::cout << "GIVEN: " << std::endl;
            next_record->getGiven()->print();
            std::cout << "EXPECTED: " << std::endl;
            for(size_t i = 0; i < next_record->getExpectedSize(); i++) {
                next_record->getExpected(i)->print();
            }

        }
    } while(next_record);

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
