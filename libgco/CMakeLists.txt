cmake_minimum_required(VERSION 2.6)

project(libgco_project)

add_library(libgco SHARED GCoptimization.cpp graph.cpp LinkedBlockList.cpp maxflow.cpp)

install(TARGETS libgco DESTINATION lib)
install(FILES LinkedBlockList.h graph.h block.h GCoptimization.h DESTINATION include)
