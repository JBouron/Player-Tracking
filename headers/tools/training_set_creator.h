#ifndef BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
#define BACHELOR_PROJECT_TRAINING_SET_CREATOR_H

#include "../background_subtractor/bgsubstractor.h"
#include "../features_comparison/feature_comparator.h"
#include "../players_extraction/player_extractor.h"
#include "../players_extraction/dpm_based_extraction/dpm_player_extractor.h"
#include "../players_extraction/blob_based_extraction/blob_player_extractor.h"
#include "../players_extraction/blob_based_extraction/blob_separator.h"

namespace tmd {
    class TrainingSetCreator {
    public:

        TrainingSetCreator(std::string video_folder, int camera_index,
                           int start_frame, int end_frame, int step_size);

        ~TrainingSetCreator();

        frame_t *next_frame();

        void set_new_video_path(std::string video_path);

        void write_centers(int frame_index);

        void write_centers();

    private:
        bool m_dpm;
        cv::VideoCapture *m_video;
        tmd::BGSubstractor *m_bgSubstractor;
        tmd::PlayerExtractor *m_playerExtractor;
        tmd::FeaturesExtractor *m_featuresExtractor;
        tmd::FeatureComparator *m_featuresComparator;
        int m_camera_index;
        std::string m_mask_path;
        int m_step;

        void extract_players_from_frame(frame_t *frame);

    };
}

#endif //BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
