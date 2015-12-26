#include "../headers/config.h"

#define load_value(name)\
    do{ \
        if (!config.lookupValue(#name, name)){ \
            tmd::debug("Config", "load_config", "Cannot load "#name", using " \
            "default value instead"); \
        } \
    }while(0)

namespace tmd{

    void Config::load_config() {
        libconfig::Config config;
        try {
            config.readFile(TMD_CONFIG_FILE);
        }
        catch (libconfig::FileIOException e){
            tmd::debug("Config", "load_config", "Can't open config file, "
                    "using default values instead.");
            return;
        }
        catch (libconfig::ParseException e){
            tmd::debug("Config", "load_config", "Invalid config file, using "
                    "default values instead.");
            return;
        }

        load_value(mask_folder);
        load_value(model_file_path);
        load_value(bgs_detect_shadows);
        load_value(bgs_threshold);
        load_value(bgs_history);
        load_value(bgs_learning_rate);
        load_value(calibration_tool_escape_char);
        load_value(dpm_detector_numthread);
        load_value(dpm_extractor_score_threshold);
        load_value(dpm_extractor_overlapping_threshold);
        load_value(dpm_extractor_duplicate_area_threshold);
        load_value(features_comparator_correlation_threshold);
        load_value(feature_extractor_threshold_red_low);
        load_value(feature_extractor_threshold_red_high);
        load_value(feature_extractor_threshold_green_low);
        load_value(feature_extractor_threshold_green_high);
        load_value(feature_extractor_threshold_saturation);
        load_value(feature_extractor_threshold_value);
        load_value(feature_extractor_histogram_size);
        load_value(sdl_binds_default_width);
        load_value(sdl_binds_default_height);
        load_value(blob_player_extractor_buffer_size);
        load_value(blob_player_extractor_min_blob_size);
        tmd::debug("Config", "load_config", "Config file loaded.");

        std::cout << "mask_folder = " << mask_folder << std::endl;
        std::cout << "bgs_detect_shadows = " <<bgs_detect_shadows  << std::endl;
        std::cout << "bgs_threshold = " <<bgs_threshold  << std::endl;
        std::cout << "bgs_history = " <<bgs_history  << std::endl;
        std::cout << "bgs_learning_rate = " <<bgs_learning_rate  << std::endl;
        std::cout << "calibration_tool_escape_char = " <<calibration_tool_escape_char  << std::endl;
        std::cout << "dpm_detector_numthread = " <<dpm_detector_numthread  << std::endl;
        std::cout << "dpm_extractor_score_threshold = " <<dpm_extractor_score_threshold  << std::endl;
        std::cout << "dpm_extractor_overlapping_threshold = " <<dpm_extractor_overlapping_threshold  << std::endl;
        std::cout << "dpm_extractor_duplicate_area_threshold = " <<dpm_extractor_duplicate_area_threshold  << std::endl;
        std::cout << "features_comparator_correlation_threshold = " <<features_comparator_correlation_threshold  << std::endl;
        std::cout << "feature_extractor_threshold_red_low = " <<feature_extractor_threshold_red_low  << std::endl;
        std::cout << "feature_extractor_threshold_red_high = " <<feature_extractor_threshold_red_high  << std::endl;
        std::cout << "feature_extractor_threshold_green_low = " <<feature_extractor_threshold_green_low  << std::endl;
        std::cout << "feature_extractor_threshold_green_high = " <<feature_extractor_threshold_green_high  << std::endl;
        std::cout << "feature_extractor_threshold_saturation = " <<feature_extractor_threshold_saturation  << std::endl;
        std::cout << "feature_extractor_threshold_value = " <<feature_extractor_threshold_value  << std::endl;
        std::cout << "feature_extractor_histogram_size = " <<feature_extractor_histogram_size  << std::endl;
        std::cout << "sdl_binds_default_width = " <<sdl_binds_default_width  << std::endl;
        std::cout << "sdl_binds_default_height = " <<sdl_binds_default_height << std::endl;
        std::cout << "blob_player_extractor_buffer_size = " <<blob_player_extractor_buffer_size  << std::endl;
        std::cout << "blob_player_extractor_min_blob_size = " <<blob_player_extractor_min_blob_size  << std::endl;
    }


    // Here are the default values in case the config file is not found or
    // invalid, or if a setting is not present in the file.

    /**********************************************************************/
    /* Misc                                                               */
    /**********************************************************************/
    std::string Config::mask_folder = "./res/bgs_masks/";
    std::string Config::model_file_path = "./res/xmls/person.xml";

    /**********************************************************************/
    /* BGS                                                                */
    /**********************************************************************/
    bool Config::bgs_detect_shadows = false;
    float Config::bgs_threshold = 256;
    int Config::bgs_history = 500;
    float Config::bgs_learning_rate = 0.0;

    /**********************************************************************/
    /* Calibration tool                                                   */
    /**********************************************************************/
    int Config::calibration_tool_escape_char = 27;

    /**********************************************************************/
    /* DPM Detector                                                       */
    /**********************************************************************/
    int Config::dpm_detector_numthread = 4;

    /**********************************************************************/
    /* DPM Player Extractor                                               */
    /**********************************************************************/
    float Config::dpm_extractor_score_threshold = -1.f;
    float Config::dpm_extractor_overlapping_threshold = 0.2;
    float Config::dpm_extractor_duplicate_area_threshold = 0.7;

    /**********************************************************************/
    /* Features Comparator                                                */
    /**********************************************************************/
    float Config::features_comparator_correlation_threshold = 0.4;

    /**********************************************************************/
    /* Features Extractor                                                 */
    /**********************************************************************/
    float Config::feature_extractor_threshold_red_low = 120;
    float Config::feature_extractor_threshold_red_high = 30;
    float Config::feature_extractor_threshold_green_low = 30;
    float Config::feature_extractor_threshold_green_high = 90;
    float Config::feature_extractor_threshold_saturation = 30;
    float Config::feature_extractor_threshold_value = 30;
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