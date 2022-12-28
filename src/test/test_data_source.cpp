//
// Created by Erik Hyrkas on 11/6/2022.
// Copyright 2022. Usable under MIT license.
//
#include <iostream>
#include "../training_data/training_dataset.hpp"
#include "../training_data/generated_datasets.hpp"

using namespace happyml;
using namespace std;

void testAdditionSource() {
    TestAdditionGeneratedDataSource testAdditionGeneratedDataSource(10);
    shared_ptr<TrainingPair> nextRecord;
    do {
        nextRecord = testAdditionGeneratedDataSource.nextRecord();
        if (nextRecord) {
            cout << "GIVEN: " << endl;
            for (const auto &given: nextRecord->getGiven()) {
                given->print();
            }
            cout << "EXPECTED: " << endl;
            for (size_t i = 0; i < nextRecord->getExpectedSize(); i++) {
                for (const auto &expected: nextRecord->getExpected()) {
                    expected->print();
                }
            }

        }
    } while (nextRecord);

//    ASSERT_TRUE(test_one_quarter(NAN, 4));
}

int main() {
    try {
        testAdditionSource();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}
