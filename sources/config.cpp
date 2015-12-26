#include "../headers/config.h"

namespace tmd{
    bool Config::bgs_detect_shadows = false;
    float Config::bgs_threshold = 256;
    int Config::bgs_history = 500;
    float Config::bgs_learning_rate = 0.0;

    int Config::calibration_tool_escape_char = 27;

    int Config::dpm_detector_numthread = 4;

    float Config::dpm_extractor_score_threshold = -1.f;
    float Config::dpm_extractor_overlapping_threshold = 0.2;
    float Config::dpm_extractor_duplicate_area_threshold = 0.7;
}