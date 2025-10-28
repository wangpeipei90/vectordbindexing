cmake_minimum_required(VERSION 3.12)
project(hnsw_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

# 查找Python和pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 使用Python模块查找pybind11（而不是系统包）
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

# 源文件
set(SOURCE_FILES
    hnsw.cpp
    hnsw_bindings.cpp
)

# 创建Python模块
pybind11_add_module(hnsw_core ${SOURCE_FILES})

# 链接库
target_link_libraries(hnsw_core PRIVATE)

# 设置输出目录
set_target_properties(hnsw_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

