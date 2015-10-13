#ifndef TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H

#include <opencv2/videoio.hpp>
#include <exception>
#include <stdexcept>
#include "frame_t.h"

#define TMD_BGS_DETECTS_SHADOWS false

namespace tmd{
    /* Class responsible of applying a BG substraction on a given video. */

    class BGSubstractor{
    public:
        BGSubstractor(cv::VideoCapture* input_video, unsigned char camera_index, float threshold = 256, int history = 500, float learning_rate = 0.0);
        ~BGSubstractor();

        bool has_next_frame();
        frame_t* next_frame();

        void set_threshold_value(float th);
        void set_history_size(int s);
        void set_learning_rate(float lr);

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgs;
        cv::VideoCapture* m_input_video;
        unsigned char m_camera_index;
        unsigned int m_frame_index;
        unsigned int m_total_frame_count;
        float m_learning_rate;
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
