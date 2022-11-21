//
// Created by Erik Hyrkas on 10/25/2022.
//

#ifndef MICROML_UNIT_TEST_HPP
#define MICROML_UNIT_TEST_HPP
#include <chrono>

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
        start_time = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        stop_time = std::chrono::high_resolution_clock::now();
    }
    void print_microseconds() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " microseconds" << std::endl;
    }

    void print_milliseconds() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " milliseconds" << std::endl;
    }

    void print_seconds() {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " seconds" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time;
};
class EvenMoreSimpleTimer {
public:
    EvenMoreSimpleTimer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void print_microseconds() {
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " microseconds" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

    void print_milliseconds() {
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " milliseconds" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

    void print_seconds() {
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
        std::cout << "Elapsed Time: " << duration.count() << " seconds" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};
#endif //MICROML_UNIT_TEST_HPP
