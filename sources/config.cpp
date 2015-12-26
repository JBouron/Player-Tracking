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

    float Config::features_comparator_correlation_threshold = 0.4;

    float Config::feature_extractor_threshold_red_low = 120;
    float Config::feature_extractor_threshold_red_high = 30;
    float Config::feature_extractor_threshold_green_low = 30;
    float Config::feature_extractor_threshold_green_high = 90;
    float Config::feature_extractor_threshold_saturation = 30;
    float Config::feature_extractor_threshold_value = 30;
    int Config::feature_extractor_histogram_size = 180;

    int Config::sdl_binds_default_width = 1376;
    int Config::sdl_binds_default_height = 992;

    int Config::blob_player_extractor_buffer_size = 5; // Must be odd
    int Config::blob_player_extractor_min_blob_size = 500;
}