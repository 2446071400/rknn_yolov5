cmake_minimum_required(VERSION 3.10)

Project(yolov5)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librga/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb_image/include)


add_executable(yolov5 main.cpp
               yolov5.cpp
               postprocess.cpp 
               ./utils/file_utils.c 
               ./utils/image_drawing.c
               ./utils/image_utils.c
               )

set(RKNN_RT_LIB /usr/lib/librknnrt.so)
set(RGA_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librga/Linux/aarch64/librga.so)
set(JPEG_LIB ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a)

target_link_libraries(yolov5
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${JPEG_LIB}
    )
