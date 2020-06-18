cmake_minimum_required(VERSION 3.5.1)
project(CarND_Extended_Kalman_Filter)

set(CMAKE_CXX_STANDARD 14)

add_executable(CarND_Extended_Kalman_Filter source/main.cpp source/FusionEKF.h source/FusionEKF.cpp source/tools.cpp source/tools.h source/kalman_filter.cpp source/kalman_filter.h)

target_link_libraries(CarND_Extended_Kalman_Filter z ssl uv uWS)