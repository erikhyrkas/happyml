cmake_minimum_required(VERSION 3.23)
project(microml)

set(CMAKE_CXX_STANDARD 17)

add_executable(microml src/main.cpp src/quarter_float.hpp src/activation.hpp src/tensor.hpp src/layer.hpp src/tensor_stats.hpp src/model.hpp src/dataset.hpp src/data_source.hpp src/optimizer.hpp src/loss.hpp src/data.hpp src/architecture.hpp src/neuron.hpp)

add_executable(microml_test_quarter_float src/test_quarter_float.cpp src/quarter_float.hpp src/unit_test.hpp)

add_executable(microml_test_quarter_matrix src/test_tensor.cpp src/quarter_float.hpp src/unit_test.hpp src/tensor.hpp src/tensor_stats.hpp)

add_executable(microml_test_data_source src/test_data_source.cpp src/data_source.hpp src/dataset.hpp src/data.hpp src/quarter_float.hpp src/unit_test.hpp src/tensor.hpp src/tensor_stats.hpp)

