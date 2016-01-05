#ifndef BACHELOR_PROJECT_FEATURES_T_H
#define BACHELOR_PROJECT_FEATURES_T_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

namespace tmd{
    /* Structure holding the features for one player. */
    typedef struct{
        std::vector<cv::Rect> body_parts;   // Boxes for each body parts.
        cv::Rect torso_pos;                 // The box of the torso.
        cv::Mat torso;                      // The torso image.
        cv::Mat torso_mask;                 // Mask for the torso.
        cv::Mat torso_color_histogram;      // Color histogram of the torso.
    }features_t;
}

#endif //BACHELOR_PROJECT_FEATURES_T_H
