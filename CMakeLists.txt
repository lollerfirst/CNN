cmake_minimum_required(VERSION 3.18)
project(CNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_compile_options(-Wall -Wextra -pedantic -Werror)
include_directories(./include ./res)
find_package(GTest CONFIG REQUIRED) 


add_subdirectory(tests)
add_subdirectory(src)
