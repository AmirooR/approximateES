cmake_minimum_required(VERSION 2.8)
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       # Mac OS X specific code
    SET(CMAKE_SYSTEM_NAME Darwin)

    # Add MacPorts
    INCLUDE_DIRECTORIES(/opt/local/include)
    LINK_DIRECTORIES(/opt/local/lib)

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

project(approximateES)
find_package( OpenCV REQUIRED )
add_subdirectory(libgco)
include_directories(libgco)
add_subdirectory(fgbg-data)
add_executable(approximateES example-gco.cpp)
add_executable(fgbg-gco fgbg-gco.cpp)

target_link_libraries(approximateES libgco ${OpenCV_LIBS})
target_link_libraries(fgbg-gco libgco ${OpenCV_LIBS})

install(TARGETS approximateES DESTINATION bin)
install(TARGETS fgbg-gco DESTINATION bin)
