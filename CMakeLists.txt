set(CMAKE_BUILD_TYPE "Release")
cmake_minimum_required(VERSION 2.8.12)
project( FeatureTracker )
find_package(depthai CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

add_executable( feature_tracker feature_tracker.cpp )
target_link_libraries( feature_tracker ${OpenCV_LIBS} depthai::core)

add_executable( visualiser visualiser.cpp )
target_link_libraries( visualiser ${OpenCV_LIBS} depthai::core)