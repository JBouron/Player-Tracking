#ifndef TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H

#include <opencv2/videoio.hpp>
#include <exception>
#include <stdexcept>
#include "frame_t.h"

#define TMD_BGS_LEARNING_RATE 0.0
#define TMD_BGS_HISTORY_SIZE 500
#define TMD_BGS_THRESHOLD 256
#define TMD_BGS_DETECTS_SHADOWS false

namespace tmd{
    /* Class responsible of applying a BG substraction on a given video. */

    class BGSubstractor{
    public:
        BGSubstractor(cv::VideoCapture* input_video, unsigned char camera_index);
        ~BGSubstractor();

        bool has_next_frame();
        frame_t* next_frame();

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgs;
        cv::VideoCapture* m_input_video;
        unsigned char m_camera_index;
        unsigned int m_frame_index;
        unsigned int m_total_frame_count;
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
