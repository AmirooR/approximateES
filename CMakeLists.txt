cmake_minimum_required(VERSION 2.8)

project(myapp)

add_subdirectory(libgco)
include_directories(libgco)
add_executable(myapp example.cpp)
target_link_libraries(myapp gco)

install(TARGETS myapp DESTINATION bin)
