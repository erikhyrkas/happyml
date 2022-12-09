cmake_minimum_required(VERSION 3.23)
project(microml)

set(CMAKE_CXX_STANDARD 17)

add_executable(microml src/quarter_float.hpp src/main.cpp src/half_float.hpp src/activation.hpp src/tensor.hpp src/tensor_stats.hpp src/model.hpp src/dataset.hpp src/optimizer.hpp src/loss.hpp src/data.hpp src/neural_network.hpp src/mbgd_optimizer.hpp src/neural_network_function.hpp src/dataencoder.hpp src/file_reader.hpp src/test/basic_profiler.hpp)

add_executable(example_xor_model src/example/example_xor_model.cpp)

add_executable(example_mnist_model src/example/example_mnist_model.cpp)

add_executable(test_quarter_float src/test/test_quarter_float.cpp)

add_executable(test_tensor src/test/test_tensor.cpp)

add_executable(test_data_source src/test/test_data_source.cpp)

add_executable(test_file_reader src/test/test_file_reader.cpp)

add_executable(test_convolution src/test/test_convolution.cpp)