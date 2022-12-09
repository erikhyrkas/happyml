//
// Created by Erik Hyrkas on 12/9/2022.
//

#ifndef MICROML_TIMERS_HPP
#define MICROML_TIMERS_HPP

#include <iostream>
#include <chrono>

using namespace std;

namespace microml {

    class ElapsedTimer {
    public:
        ElapsedTimer() {
            start_time = std::chrono::high_resolution_clock::now();
        }

        long long int getMicroseconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getMilliseconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getSeconds() {
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
            start_time = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    };

    class SimpleTimer {
    public:
        SimpleTimer() = default;

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

}
#endif //MICROML_TIMERS_HPP
