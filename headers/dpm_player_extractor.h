#ifndef BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H

#include <opencv2/objdetect/objdetect.hpp>
#include "player_extractor.h"

#define TMD_DEFAULT_DMP_EXTRACTOR_SCORE_THRESHOLD -1.0
#define TMD_DEFAULT_DMP_EXTRACTOR_OVERLAPPING_THRESHOLD 0.2

// Percentage of shared area such that it is considered as a dupluicate.
#define TMD_DPM_EXTRACTOR_DUPLICATE_AREA_THRESHOLD 0.7 // %


namespace tmd{

    /**
     * Player extractor using DPM to detect and crop players from the frame.
     */
    class DPMPlayerExtractor : public tmd::PlayerExtractor{
    public :
        /**
         * Constructor of the DPMPlayerExtractor
         * _ model_file : path to the file containing the model of a person.
         * _ overlap_threshold : The overlap threshold to use with DPM.
         * _ score_threshold : The score threshold to use to filter results.
         */
        DPMPlayerExtractor(std::string model_file, float overlap_threshold =
        TMD_DEFAULT_DMP_EXTRACTOR_OVERLAPPING_THRESHOLD, float score_threshold
        = TMD_DEFAULT_DMP_EXTRACTOR_SCORE_THRESHOLD);

        ~DPMPlayerExtractor();

        /**
         * Extract a player from a given frame. Returns a vector containing
         * the detected players.
         */
        virtual std::vector<player_t*> extract_player_from_frame(frame_t*
        frame);

        /**
         * Setters for the 2 thresholds.
         */
        void set_overlapping_threshold(float th);
        void set_score_threshold(float th);

        /**
         * Getters for the 2 thresholds.
         */
        float get_overlapping_threshold();
        float get_score_threshold();

    private:

        /**
         * Returns the "colored mask", ie all the pixels belonging to the
         * background are in black, whereas the others are in color. The
         * resulting image are colored forground objects on a black background.
         * The point of doing this is to improve the results of dpm.
         */
        cv::Mat get_colored_mask_for_frame(frame_t* frame);

        cv::LatentSvmDetector* m_detector;
        float m_overlap_threshold;
        float m_score_threshold;
    };
}

#endif //BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H
