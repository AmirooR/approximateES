cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

#IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
IF(APPLE)
       # Mac OS X specific code
    SET(CMAKE_SYSTEM_NAME Darwin)

    # Add MacPorts
    INCLUDE_DIRECTORIES(/opt/local/include)
    LINK_DIRECTORIES(/opt/local/lib)
ENDIF(APPLE)
#ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

project(approximateES)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package( OpenCV REQUIRED )
add_subdirectory(libgco)
include_directories(libgco)
add_subdirectory(fgbg-data)
add_subdirectory(fgbg-spatialy-varying-smooth)
add_subdirectory(dense)
add_executable(approximateES example-gco.cpp)
add_executable(fgbg-gco fgbg-gco.cpp)
add_executable(stereo-orig stereo-orig.cpp)

target_link_libraries(stereo-orig libgco ${OpenCV_LIBS})
target_link_libraries(approximateES libgco ${OpenCV_LIBS})
target_link_libraries(fgbg-gco libgco ${OpenCV_LIBS})

install(TARGETS approximateES DESTINATION bin)
install(TARGETS fgbg-gco DESTINATION bin)
