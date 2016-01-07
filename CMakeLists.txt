cmake_minimum_required(VERSION 3.3)
project(Bachelor_Project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14 -D VALGRIND=ON -static-libstdc++ -static-libgcc -I /home/jbouron/Downloads/boost_1_60_0/")

set(BUILD_SHARED_LIBS OFF)
set(OpenCV_FOUND TRUE)
find_package(OpenCV REQUIRED)

ADD_LIBRARY(libconfig STATIC IMPORTED)
SET_TARGET_PROPERTIES(libconfig PROPERTIES IMPORTED_LOCATION /usr/local/lib/libconfig++.a)

INCLUDE(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

INCLUDE_DIRECTORIES(${SDL2_INCLUDE_DIRS})

set(SOURCE_FILES sources/main.cpp
        headers/data_structures/frame_t.h
        headers/data_structures/player_t.h
        headers/data_structures/team_t.h
        headers/background_subtractor/bgsubstractor.h
        sources/background_subtractor/bgsubstractor.cpp
        headers/misc/debug.h
        headers/players_extraction/player_extractor.h
        headers/features_extraction/features_extractor.h
        headers/tools/calibration_tool.h
        sources/tools/calibration_tool.cpp
        headers/data_structures/features_t.h
        headers/tools/manual_player_extractor.h
        sources/tools/manual_player_extractor.cpp
        headers/features_comparison/feature_comparator.h
        sources/features_comparison/feature_comparator.cpp
        headers/openCV/_latentsvm.h
        headers/openCV/_lsvm_distancetransform.h
        headers/openCV/_lsvm_error.h
        headers/openCV/_lsvm_fft.h
        headers/openCV/_lsvm_matching.h
        headers/openCV/_lsvm_routine.h
        headers/openCV/_lsvm_types.h
        sources/features_extraction/features_extractor.cpp
        headers/players_extraction/dpm_based_extraction/dpm_player_extractor.h
        sources/players_extraction/dpm_based_extraction/dpm_player_extractor.cpp
        headers/tools/dpm_calibrator.h
        sources/tools/dpm_calibrator.cpp
        headers/players_extraction/blob_based_extraction/blob_player_extractor.h
        sources/players_extraction/blob_based_extraction/blob_player_extractor.cpp
        headers/pipelines/pipeline.h
        sources/pipelines/pipeline.cpp
        sources/tools/training_set_creator.cpp
        headers/tools/training_set_creator.h
        headers/players_extraction/blob_based_extraction/blob_separator.h
        sources/players_extraction/blob_based_extraction/blob_separator.cpp
        headers/sdl_binds/sdl_binds.h
        sources/sdl_binds/sdl_binds.cpp
        headers/pipelines/simple_pipeline.h
        sources/pipelines/simple_pipeline.cpp
        headers/pipelines/pipeline_thread.h
        sources/pipelines/pipeline_thread.cpp
        headers/pipelines/multithreaded_pipeline.h
        sources/pipelines/multithreaded_pipeline.cpp
        headers/misc/config.h sources/misc/config.cpp
        headers/features_extraction/dpm.h
        sources/features_extraction/dpm.cpp
        headers/pipelines/approximative_pipeline.h
        sources/pipelines/approximative_pipeline.cpp
        headers/data_structures/cmd_args_t.h)

add_executable(Bachelor_Project ${SOURCE_FILES})
target_link_libraries(Bachelor_Project ${OpenCV_LIBS} ${SDL2_LIBRARIES}
        libconfig
)
