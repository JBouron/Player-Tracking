cmake_minimum_required(VERSION 3.3)
project(Bachelor_Project)

# Note : -MM and -H can only build, not run the program.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -D VALGRIND=ON")
set(OpenCV_FOUND TRUE)
find_package(OpenCV REQUIRED)
set(SOURCE_FILES sources/main.cpp headers/frame_t.h headers/player_t.h
        headers/team_t.h headers/position_t.h headers/box_t.h
        headers/bgsubstractor.h sources/bgsubstractor.cpp headers/debug.h
        headers/player_extractor.h headers/features_extractor.h
        headers/coordinates_translator.h sources/coordinates_translator.cpp
        headers/calibration_tool.h sources/calibration_tool.cpp
        headers/features_t.h headers/log_parser.h sources/log_parser.cpp
        headers/manual_player_extractor.h sources/manual_player_extractor.cpp
        headers/dpm_detector.h sources/dpm_detector.cpp
        headers/feature_comparator.h sources/feature_comparator.cpp
        headers/openCV/_latentsvm.h headers/openCV/_lsvm_distancetransform.h
        headers/openCV/_lsvm_error.h headers/openCV/_lsvm_fft.h
        headers/openCV/_lsvm_matching.h headers/openCV/_lsvm_routine.h
        headers/openCV/_lsvm_types.h sources/features_extractor.cpp
        headers/test_cases/bgsubstractor_tests.h
        sources/test_cases/bgsubstractor_tests.cpp
        headers/test_cases/test_suite.h sources/test_cases/test_suite.cpp
        headers/test_cases/dpm_detector_tests.h
        sources/test_cases/dpm_detector_tests.cpp
        headers/test_cases/features_extractor_tests.h
        sources/test_cases/features_extractor_tests.cpp
        headers/demo/demo.h
        sources/demo/demo.cpp
        headers/test_cases/feature_comparator_tests.h
        sources/test_cases/feature_comprator_tests.cpp
        headers/dpm_player_extractor.h
        sources/dpm_player_extractor.cpp
        headers/dpm_calibrator.h
        sources/dpm_calibrator.cpp
        headers/blob_player_extractor.h
        sources/blob_player_extractor.cpp
        headers/pipeline.h
        sources/pipeline.cpp
        sources/training_set_creator.cpp
        headers/training_set_creator.h
        headers/blob_separator.h
        sources/blob_separator.cpp
        headers/pipeline_visualizer.h sources/pipeline_visualizer.cpp)
add_executable(Bachelor_Project ${SOURCE_FILES})
target_link_libraries(Bachelor_Project ${OpenCV_LIBS} cppunit)
