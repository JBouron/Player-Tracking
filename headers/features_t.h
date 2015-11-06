#ifndef BACHELOR_PROJECT_FEATURES_T_H
#define BACHELOR_PROJECT_FEATURES_T_H


namespace tmd{
    /* Structure holding the features for one player. */
    typedef struct{
        std::vector<cv::Rect> body_parts;
        cv::Mat color_histogram;
    }features_t;
}

#endif //BACHELOR_PROJECT_FEATURES_T_H
