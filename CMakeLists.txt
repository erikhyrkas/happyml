cmake_minimum_required(VERSION 3.23)
project(microml)

set(CMAKE_CXX_STANDARD 17)

add_executable(microml src/main.cpp src/quarter_float.hpp src/activation.hpp src/tensor.hpp src/tensor_stats.hpp src/model.hpp src/dataset.hpp src/optimizer.hpp src/loss.hpp src/data.hpp src/neural_network.hpp src/sgd_optimizer.hpp src/neural_network_function.hpp src/dataencoder.hpp src/file_reader.hpp)

add_executable(microml_test_quarter_float src/test_quarter_float.cpp src/quarter_float.hpp src/unit_test.hpp src/neural_network.hpp)

add_executable(microml_test_tensor src/test_tensor.cpp src/quarter_float.hpp src/unit_test.hpp src/tensor.hpp src/tensor_stats.hpp src/neural_network.hpp)

add_executable(microml_test_data_source src/test_data_source.cpp src/dataset.hpp src/data.hpp src/quarter_float.hpp src/unit_test.hpp src/tensor.hpp src/tensor_stats.hpp src/neural_network.hpp)

add_executable(microml_test_xor_model src/test_xor_model.cpp)

add_executable(microml_test_file_reader src/test_file_reader.cpp)