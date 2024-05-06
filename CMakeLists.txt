cmake_minimum_required(VERSION 3.16)
project(GNNAlgorithm)
include(../../common.cmake)

add_executable(gnnAlgorithm gnnAlgorithm.cpp)
target_link_libraries(gnnAlgorithm Kokkos::kokkos)