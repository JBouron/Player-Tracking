#ifndef BACHELOR_PROJECT_FEATURES_T_H
#define BACHELOR_PROJECT_FEATURES_T_H


namespace tmd{
    /* Structure holding the features for one player. */
    typedef struct{
        std::vector<cv::Rect> body_parts;   // Boxes for each body parts.
        cv::Rect torso;                     // Box for the torso.
        cv::Mat torso_mask;                 // Mask for the torso.
        cv::Mat torso_color_histogram;      // Color histogram of the torso.
    }features_t;
}

#endif //BACHELOR_PROJECT_FEATURES_T_H
