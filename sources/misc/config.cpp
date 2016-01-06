#include "../../headers/misc/config.h"

using namespace std;

/**
 * The standard library does not define the std::to_string(std::string)
 * method, so we made our own.
 */
string to_string(string str){
    return str;
}

/**
 * Useful macro allowing us to load each field and display its value.
 */
#define load_value(name)\
    do{ \
        config.lookupValue(#name, name); \
            tmd::debug("Config", "load_config","" #name " = " + \
            to_string(name)); \
    }while(0) // The while (0) is an old trick to make the semi colon
                //mandatory after calling the macro.

namespace tmd {

    void Config::load_config() {
        libconfig::Config config;
        try {
            config.readFile(TMD_CONFIG_FILE);
        }
        catch (libconfig::FileIOException e) {
            tmd::debug("Config", "load_config", "Can't open config file, "
                    "using default values instead.");
            return;
        }
        catch (libconfig::ParseException e) {
            tmd::debug("Config", "load_config", "Invalid config file, using "
                    "default values instead.");
            return;
        }

        //load_value(mask_folder);
        //load_value(model_file_path);
        load_value(bgs_detect_shadows);
        load_value(bgs_threshold);
        load_value(bgs_history);
        load_value(bgs_blob_buffer_size);
        load_value(bgs_blob_threshold_count);
        load_value(bgs_learning_rate);
        //load_value(bgs_empty_room_background);
        //load_value(calibration_tool_escape_char);
        load_value(dpm_detector_numthread);
        load_value(dpm_extractor_score_threshold);
        load_value(dpm_extractor_overlapping_threshold);
        //load_value(dpm_extractor_duplicate_area_threshold);
        load_value(features_comparator_correlation_threshold);
        //load_value(features_comparator_center_count);
        //load_value(features_comparator_sample_cols);
        //load_value(features_comparator_centers_file_rows);
        //load_value(features_comparator_centers_file_cols);
        load_value(features_comparator_centers_file_name);
        //load_value(feature_extractor_histogram_size);
        //load_value(sdl_binds_default_width);
        //load_value(sdl_binds_default_height);
        load_value(blob_player_extractor_buffer_size);
        load_value(blob_player_extractor_min_blob_size);
        load_value(use_bgs);
        load_value(use_dpm_player_extractor);
        load_value(use_colored_mask_in_dpm);
        load_value(draw_static_boxes);
        load_value(static_boxes_width);
        load_value(static_boxes_height);
        load_value(static_boxes_thickness);
        load_value(show_torsos);
        load_value(show_body_parts);
        load_value(show_players);
        load_value(show_results);
        load_value(save_results);
        load_value(show_blobs);
        load_value(show_player_team);
        load_value(save_all_frames);

        tmd::debug("Config", "load_config", "Config file loaded.");
    }


    // Here are the default values in case the config file is not found or
    // invalid, or if a setting is not present in the file.

    /**********************************************************************/
    /* Misc                                                               */
    /**********************************************************************/
    std::string Config::mask_folder = "./res/bgs_masks/";
    std::string Config::model_file_path = "./res/xmls/person.xml";
    bool Config::draw_static_boxes = true;
    int Config::static_boxes_width = 30;
    int Config::static_boxes_height = 120;
    int Config::static_boxes_thickness = 2;

    /**********************************************************************/
    /* Results display                                                    */
    /**********************************************************************/
    bool Config::show_results = false;
    bool Config::save_results = true;
    bool Config::show_torsos = false;
    bool Config::show_body_parts = false;
    bool Config::show_players = true;
    bool Config::show_blobs = false;
    bool Config::show_player_team = true;
    bool Config::save_all_frames = false;

    /**********************************************************************/
    /* BGS                                                                */
    /**********************************************************************/
    bool Config::bgs_detect_shadows = false;
    float Config::bgs_threshold = 256;
    int Config::bgs_history = 500;
    float Config::bgs_learning_rate = 0.0;
    int Config::bgs_blob_buffer_size = 2;
    int Config::bgs_blob_threshold_count = 5;
    std::string Config::bgs_empty_room_background = "./res/room_background/";
    bool Config::use_bgs = true;

    /**********************************************************************/
    /* Calibration tool                                                   */
    /**********************************************************************/
    int Config::calibration_tool_escape_char = 27;

    /**********************************************************************/
    /* DPM Detector                                                       */
    /**********************************************************************/
    int Config::dpm_detector_numthread = 4;

    /**********************************************************************/
    /* DPM                                                                */
    /**********************************************************************/
    float Config::dpm_extractor_score_threshold = -1.f;
    float Config::dpm_extractor_overlapping_threshold = 0.1;
    float Config::dpm_extractor_duplicate_area_threshold = 0.7;
    bool Config::use_dpm_player_extractor = false;
    bool Config::use_colored_mask_in_dpm = true;

    /**********************************************************************/
    /* Features Comparator                                                */
    /**********************************************************************/
    float Config::features_comparator_correlation_threshold = 0.0;
    int Config::features_comparator_center_count = 2;
    int Config::features_comparator_sample_cols = 180;
    int Config::features_comparator_centers_file_rows = 2;
    int Config::features_comparator_centers_file_cols = 180;
    std::string Config::features_comparator_centers_file_name =
            "./res/cluster/cluster.kfc";

    /**********************************************************************/
    /* Features Extractor                                                 */
    /**********************************************************************/
    int Config::feature_extractor_histogram_size = 180;

    /**********************************************************************/
    /* SDL Binds                                                          */
    /**********************************************************************/
    int Config::sdl_binds_default_width = 1376;
    int Config::sdl_binds_default_height = 992;

    /**********************************************************************/
    /* Blob Player Extractor                                              */
    /**********************************************************************/
    int Config::blob_player_extractor_buffer_size = 5; // Must be odd
    int Config::blob_player_extractor_min_blob_size = 500;
}