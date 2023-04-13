cmake_minimum_required(VERSION 3.23)
project(happyml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(happyml src/main.cpp)

# examples of using happyml directly without using the scripting language:
add_executable(example_xor_model_tanh src/example/example_xor_model_tanh.cpp)

add_executable(example_xor_model_relu src/example/example_xor_model_relu.cpp)

add_executable(example_xor_model_sigmoid src/example/example_xor_model_sigmoid.cpp)

add_executable(example_mnist_model_convolution src/example/example_mnist_model_convolution.cpp)

add_executable(example_mnist_model_8bit src/example/example_mnist_model_8bit.cpp)

add_executable(example_mnist_model_full src/example/example_mnist_model_full.cpp)

add_executable(example_bpe_model src/example/example_bpe_model.cpp)

# TODO: Need to learn how to use CTest and convert the tests to use it.
#  https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html
#  What you see below is me doing what I knew would work for local testing.
#  Keep in mind, I work in a lot of programing languages and sometimes I'm rusty or
#  don't keep up with changes... or never learned about it from the beginning.
#  I'm of the opinion that it's better to do a whole project with what you know rather
#  than do a small portion of a project perfectly and never finish. I will eventually
#  circle back and take what I learned to do this better.
add_executable(test_quarter_float src/test/test_quarter_float.cpp)

add_executable(test_tensor src/test/test_tensor.cpp)

add_executable(test_data_source src/test/test_data_source.cpp)

add_executable(test_file_reader src/test/test_file_reader.cpp)

add_executable(test_convolution src/test/test_convolution.cpp)

add_executable(test_activation src/test/test_activation.cpp)

add_executable(test_portable_bytes src/test/test_portable_bytes.cpp)

add_executable(test_lexer src/test/test_lexer.cpp)

add_executable(test_interpreter src/test/test_interpreter.cpp)

add_executable(test_byte_pair_encoding src/test/test_byte_pair_encoding.cpp)

add_executable(test_rotary_positional_embedding src/test/test_rotary_positional_embedding.cpp)

add_executable(test_one_hot_encoding src/test/test_one_hot_encoding.cpp)

add_executable(test_create_dataset_statement src/test/test_create_dataset_statement.cpp)