//
// Created by Erik Hyrkas on 10/25/2022.
//

#ifndef MICROML_UNIT_TEST_HPP
#define MICROML_UNIT_TEST_HPP

#include <iostream>

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


#endif //MICROML_UNIT_TEST_HPP
