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
            startTime = std::chrono::high_resolution_clock::now();
        }

        long long int getMicroseconds() {
            auto stopTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime);
            startTime = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getMilliseconds() {
            auto stopTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
            startTime = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

        long long int getSeconds() {
            auto stopTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(stopTime - startTime);
            startTime = std::chrono::high_resolution_clock::now();
            return duration.count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    };

    class SimpleTimer {
    public:
        SimpleTimer() = default;

        void start() {
            startTime = chrono::high_resolution_clock::now();
        }

        void stop() {
            stopTime = chrono::high_resolution_clock::now();
        }

        void printMicroseconds() {
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
        }

        void printMilliseconds() {
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
        }

        void printSeconds() {
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> startTime;
        chrono::time_point<chrono::high_resolution_clock> stopTime;
    };

    class EvenMoreSimpleTimer {
    public:
        EvenMoreSimpleTimer() {
            startTime = chrono::high_resolution_clock::now();
        }

        void printMicroseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

        void printMilliseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

        void printSeconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> startTime;
    };

}
#endif //MICROML_TIMERS_HPP
