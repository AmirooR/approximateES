cmake_minimum_required(VERSION 2.6)

project(libgco_project)

add_library(gco SHARED GCoptimization.cpp graph.cpp LinkedBlockList.cpp maxflow.cpp)

install(TARGETS gco DESTINATION lib)
install(FILES LinkedBlockList.h graph.h block.h GCoptimization.h DESTINATION include)
