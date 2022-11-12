cmake_minimum_required(VERSION 3.18)
project(CNN)

add_compile_options(-Wall -Wextra -pedantic -Werror -std=c++20)
include_directories(./include ./res)
find_package(GTest CONFIG REQUIRED) 