//
// Created by nicolas on 28.11.15.
//

#ifndef BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
#define BACHELOR_PROJECT_TRAINING_SET_CREATOR_H

#include "bgsubstractor.h"
#include "feature_comparator.h"
#include "player_extractor.h"

namespace tmd {
    class TrainingSetCreator {
    public:
        TrainingSetCreator(std::string video_path, std::string static_mask_path, int camera_index,
                           std::string model_file, bool dpm = false, bool save_frames = false,
                           std::string output_folder = "");

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

    };
}

#endif //BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
