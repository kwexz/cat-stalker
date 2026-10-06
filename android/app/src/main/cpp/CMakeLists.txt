cmake_minimum_required(VERSION 3.22.1)
project(cat_stalker_ncnn)

add_library(cat_stalker_ncnn SHARED ncnn_detector.cpp)
target_include_directories(cat_stalker_ncnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/ncnn)
add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libncnn.so)
target_link_libraries(cat_stalker_ncnn ncnn android jnigraphics log)
