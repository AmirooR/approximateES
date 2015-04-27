cmake_minimum_required(VERSION 2.8)

project(approximateES)
find_package( OpenCV REQUIRED )
add_subdirectory(libgco)
include_directories(libgco)
add_executable(approximateES example-gco.cpp)
target_link_libraries(approximateES libgco ${OpenCV_LIBS})

install(TARGETS approximateES DESTINATION bin)
