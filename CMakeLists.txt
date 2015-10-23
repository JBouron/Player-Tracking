cmake_minimum_required(VERSION 3.3)
project(Bachelor_Project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(OpenCV_FOUND TRUE)
find_package( OpenCV REQUIRED )
set(SOURCE_FILES sources/main.cpp headers/frame_t.h headers/player_t.h headers/team_t.h headers/position_t.h headers/box_t.h headers/bgsubstractor.h sources/bgsubstractor.cpp headers/debug.h sources/debug.cpp headers/player_extractor.h headers/features_extractor.h headers/coordinates_translator.h sources/coordinates_translator.cpp headers/calibration_tool.h sources/calibration_tool.cpp headers/features_t.h headers/log_parser.h sources/log_parser.cpp tests/heuristic_feature_extractor_tests.cpp sources/heuristic_feature_extractor.cpp headers/manual_player_extractor.h sources/manual_player_extractor.cpp)
add_executable(Bachelor_Project ${SOURCE_FILES})
target_link_libraries(Bachelor_Project ${OpenCV_LIBS})