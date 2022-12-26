cmake_minimum_required(VERSION 3.23)
project(happyml)

set(CMAKE_CXX_STANDARD 17)

add_executable(happyml src/main.cpp)

add_executable(example_xor_model_tanh src/example/example_xor_model_tanh.cpp)

add_executable(example_xor_model_relu src/example/example_xor_model_relu.cpp)

add_executable(example_xor_model_sigmoid src/example/example_xor_model_sigmoid.cpp)

add_executable(example_mnist_model_convolution src/example/example_mnist_model_convolution.cpp)

add_executable(example_mnist_model_8bit src/example/example_mnist_model_8bit.cpp)

add_executable(example_mnist_model_full src/example/example_mnist_model_full.cpp)

add_executable(test_quarter_float src/test/test_quarter_float.cpp)

add_executable(test_tensor src/test/test_tensor.cpp)

add_executable(test_data_source src/test/test_data_source.cpp)

add_executable(test_file_reader src/test/test_file_reader.cpp)

add_executable(test_convolution src/test/test_convolution.cpp)

add_executable(test_activation src/test/test_activation.cpp)

add_executable(test_portable_bytes src/test/test_portable_bytes.cpp)