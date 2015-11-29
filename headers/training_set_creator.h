//
// Created by nicolas on 28.11.15.
//

#ifndef BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
#define BACHELOR_PROJECT_TRAINING_SET_CREATOR_H

#include "bgsubstractor.h"

namespace tmd {
    class TrainingSetCreator {
    public:
        TrainingSetCreator(std::string* video_path, std::string static_mask_path, unsigned char camera_index,
                           std::string model_file, bool dpm
        = false, bool save_frames = false,
                           std::string output_folder = "");

        ~TrainingSetCreator();


    };
}

#endif //BACHELOR_PROJECT_TRAINING_SET_CREATOR_H
