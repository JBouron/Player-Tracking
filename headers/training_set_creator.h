#ifndef BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
#define BACHELOR_PROJECT_TRAINING_SET_CREATOR_H

#include "bgsubstractor.h"
#include "feature_comparator.h"
#include "player_extractor.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/blob_separator.h"

namespace tmd {
    class TrainingSetCreator {
    public:

        TrainingSetCreator(std::string video_folder, int camera_index, std::string model_file, int start_frame,
                           int end_frame, int step_sizer);

        ~TrainingSetCreator();

        frame_t *next_frame();

        void set_new_video_path(std::string video_path);

        void write_centers();

        void set_frame_step_size(int step);

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
