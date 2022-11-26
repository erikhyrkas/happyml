//
// Created by Erik Hyrkas on 10/25/2022.
//

#ifndef MICROML_UNIT_TEST_HPP
#define MICROML_UNIT_TEST_HPP
#include <chrono>

using namespace std;

#define ASSERT_TRUE(arg) \
            if(!(arg)) { \
                std::cout << "Test failed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl; \
               throw std::exception("Test failed."); \
            } \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl

#define ASSERT_FALSE(arg) \
            if((arg)) { \
                std::cout << "Test failed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl; \
               throw std::exception("Test failed."); \
            }             \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl


class SimpleTimer {
public:
    SimpleTimer() {

    }
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }
    void stop() {
        stop_time = chrono::high_resolution_clock::now();
    }
    void print_microseconds() {
        auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
    }

    void print_milliseconds() {
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
    }

    void print_seconds() {
        auto duration = chrono::duration_cast<chrono::seconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
    }

private:
    chrono::time_point<chrono::high_resolution_clock> start_time;
    chrono::time_point<chrono::high_resolution_clock> stop_time;
};
class EvenMoreSimpleTimer {
public:
    EvenMoreSimpleTimer() {
        start_time = chrono::high_resolution_clock::now();
    }

    void print_microseconds() {
        auto stop_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
        start_time = chrono::high_resolution_clock::now();
    }

    void print_milliseconds() {
        auto stop_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
        start_time = chrono::high_resolution_clock::now();
    }

    void print_seconds() {
        auto stop_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(stop_time - start_time);
        cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
        start_time = chrono::high_resolution_clock::now();
    }

private:
    chrono::time_point<chrono::high_resolution_clock> start_time;
};
#endif //MICROML_UNIT_TEST_HPP
