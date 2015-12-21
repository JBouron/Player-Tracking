#ifndef BACHELOR_PROJECT_FEATURES_T_H
#define BACHELOR_PROJECT_FEATURES_T_H


namespace tmd{
    /* Structure holding the features for one player. */
    typedef struct{
        std::vector<cv::Rect> body_parts;   // Boxes for each body parts.
        cv::Rect torso_pos;
        cv::Mat torso;                     // Box for the torso.
        cv::Mat torso_mask;                 // Mask for the torso.
        cv::Mat torso_color_histogram;      // Color histogram of the torso.
    }features_t;

    inline void free_features(features_t* features){
        features->torso.release();
        features->torso_mask.release();
        features->torso_color_histogram.release();
    }
}

#endif //BACHELOR_PROJECT_FEATURES_T_H
