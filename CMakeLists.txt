cmake_minimum_required(VERSION 3.3)
project(Bachelor_Project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(OpenCV_FOUND TRUE)
find_package( OpenCV REQUIRED )
set(SOURCE_FILES sources/main.cpp headers/frame_t.h headers/player_t.h headers/team_t.h headers/position_t.h headers/box_t.h headers/bgsubstractor.h sources/bgsubstractor.cpp headers/debug.h sources/debug.cpp sources/player_extractor.cpp headers/player_extractor.h sources/features_extractor.cpp headers/features_extractor.h headers/coordinates_translator.h sources/coordinates_translator.cpp)
add_executable(Bachelor_Project ${SOURCE_FILES})
target_link_libraries(Bachelor_Project ${OpenCV_LIBS})