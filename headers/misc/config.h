#ifndef BACHELOR_PROJECT_CONFIG_H
#define BACHELOR_PROJECT_CONFIG_H

#include <libconfig.h++>
#include "debug.h"

#define TMD_CONFIG_FILE "./config.cfg"

namespace tmd{
    /**
     * Class representing the config of the program.
     * This is the memory representation of the configuration file config.cfg.
     *
     * Every member of this class can be redefined in the configuration file
     * using the same name.
     *
     * All members have a default value. When a member is found in the
     * configuration file, its value is overwritten by the one found in the
     * file.
     */
    class Config{
    public:
        /**
         * Load the config from the config file.
         */
        static void load_config();

        /**
         * Here are all configs / parameters / values ...
         * They are public for ease of use.
         */

        /**********************************************************************/
        /* Misc                                                               */
        /**********************************************************************/
        static std::string mask_folder;
        static std::string model_file_path;
        static bool draw_static_boxes;
        static int static_boxes_width;
        static int static_boxes_height;
        static int static_boxes_thickness;

        /**********************************************************************/
        /* Results display                                                    */
        /**********************************************************************/
        static bool show_results;
        static bool save_results;
        static bool show_torsos;
        static bool show_body_parts;
        static bool show_players;
        static bool show_blobs;
        static bool show_player_team;

        /**********************************************************************/
        /* BGS                                                                */
        /**********************************************************************/
        static bool bgs_detect_shadows;
        static float bgs_threshold;
        static int bgs_history;
        static float bgs_learning_rate;
        static int bgs_blob_buffer_size;
        static int bgs_blob_threshold_count;
        static std::string bgs_empty_room_background;
        static bool use_bgs;

        /**********************************************************************/
        /* Calibration tool                                                   */
        /**********************************************************************/
        static int calibration_tool_escape_char;

        /**********************************************************************/
        /* DPM Detector                                                       */
        /**********************************************************************/
        static int dpm_detector_numthread;

        /**********************************************************************/
        /* DPM                                                                */
        /**********************************************************************/
        static float dpm_extractor_score_threshold;
        static float dpm_extractor_overlapping_threshold;
        static float dpm_extractor_duplicate_area_threshold;
        static bool use_dpm_player_extractor;
        static bool use_colored_mask_in_dpm;

        /**********************************************************************/
        /* Features Comparator                                                */
        /**********************************************************************/
        static float features_comparator_correlation_threshold;
        static int features_comparator_center_count;
        static int features_comparator_sample_cols;
        static int features_comparator_centers_file_rows;
        static int features_comparator_centers_file_cols;
        static std::string features_comparator_centers_file_name;

        /**********************************************************************/
        /* Features Extractor                                                 */
        /**********************************************************************/
        static float feature_extractor_threshold_red_low;
        static float feature_extractor_threshold_red_high;
        static float feature_extractor_threshold_green_low;
        static float feature_extractor_threshold_green_high;
        static float feature_extractor_threshold_saturation;
        static float feature_extractor_threshold_value;
        static int feature_extractor_histogram_size;

        /**********************************************************************/
        /* SDL Binds                                                          */
        /**********************************************************************/
        static int sdl_binds_default_width;
        static int sdl_binds_default_height;

        /**********************************************************************/
        /* Blob Player Extractor                                              */
        /**********************************************************************/
        static int blob_player_extractor_buffer_size;
        static int blob_player_extractor_min_blob_size;
    };
}

#endif //BACHELOR_PROJECT_CONFIG_H
