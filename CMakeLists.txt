cmake_minimum_required(VERSION 3.12)
project(hello)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#set(CMAKE_LINK_WHAT_YOU_USE TRUE)
#find_program(TIDY "clang-tidy-6.0")
#set(CMAKE_CXX_CLANG_TIDY ${TIDY} "-checks=*")
#find_program(CPPCHECK cppcheck "--std=c++14")
#set(CMAKE_CXX_CPPCHECK ${CPPCHECK})
#find_program(IWYU include-what-you-use)
#set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE  ${IWYU})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(OpenCV_DIR "/home/afterburner/libraries/opencv4_release/lib/cmake/opencv4")
    message(STATUS "using opencv release")
endif()

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED CONFIG)

add_library(segmentation STATIC segmentation.cpp)
target_link_libraries(segmentation PRIVATE  opencv_imgproc opencv_core opencv_highgui fmt::fmt)

add_library(rotation STATIC rotate.cpp)
target_link_libraries(rotation PRIVATE  opencv_imgproc opencv_core opencv_highgui fmt::fmt)

add_library(stretch STATIC stretch.cpp)
target_link_libraries(stretch PRIVATE  opencv_imgproc opencv_core opencv_highgui fmt::fmt)

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE
        opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui fmt::fmt
        segmentation rotation stretch
        )
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(main PRIVATE DEBUG)
endif()
