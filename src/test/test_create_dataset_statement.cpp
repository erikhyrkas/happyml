//
// Created by Erik Hyrkas on 4/13/2023.
//

#include "../util/timers.hpp"
#include "../lang//statements.hpp"

using namespace std;
using namespace happyml;

void test_multi_input_multi_output_create() {
    CreateDatasetStatement test();
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