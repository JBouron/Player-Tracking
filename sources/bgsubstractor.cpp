#include <opencv2/core/core.hpp>
#include "../headers/bgsubstractor.h"
#include "../headers/debug.h"
#include "../headers/frame_t.h"

namespace tmd{
    BGSubstractor::BGSubstractor(cv::VideoCapture* input_video, unsigned char camera_index, float threshold, int history, float learning_rate){
        m_bgs = new cv::BackgroundSubtractorMOG2(history, threshold, TMD_BGS_DETECTS_SHADOWS);
        if (m_bgs == NULL){
            throw std::bad_alloc();
        }
        m_learning_rate = learning_rate;
        tmd::debug("BGSubstractor", "BGSubstractor", "bgs created.");

        m_input_video = input_video;
        if (m_input_video == NULL || !m_input_video->isOpened()){
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "input video is not valid (NULL or not opened).");
        }

        tmd::debug("BGSubstractor", "BGSubstractor", "valid input video.");
        m_camera_index = camera_index;
        if (!(0 <= m_camera_index && m_camera_index < 8)){
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "invalid camera index " + m_camera_index);
        }
        tmd::debug("BGSubstractor", "BGSubstractor", "valid camera index");
        m_frame_index = 0;
        m_total_frame_count = static_cast<unsigned int> (m_input_video->get(CV_CAP_PROP_FRAME_COUNT));
        tmd::debug("BGSubstractor", "BGSubstractor", "exiting method");
    }

    BGSubstractor::~BGSubstractor(){
        free(m_bgs);
    }

    bool BGSubstractor::has_next_frame(){
        return m_frame_index != m_total_frame_count;
    }

    frame_t* BGSubstractor::next_frame(){
        if (!this->has_next_frame()){
            throw std::runtime_error("Error in BGSubstractor::next_frame() : no frame left.");
        }
        frame_t* frame = new frame_t;
        m_input_video->read(frame->original_frame);
        frame->frame_index = m_frame_index;
        cv::Mat mask;
        m_bgs->operator()(frame->original_frame, frame->mask_frame, m_learning_rate);
        frame->camera_index = m_camera_index;

        m_frame_index ++;
        tmd::debug("BGSubstractor", "next_frame", "frame fetched. Frame index = " +
                std::to_string(m_frame_index));

        return frame;
    }

    void BGSubstractor::set_threshold_value(float th){
        m_bgs->set("varThreshold", th);
    }

    void BGSubstractor::set_history_size(int s){
        m_bgs->set("history", s);
    }

    void BGSubstractor::set_learning_rate(float lr){
        m_learning_rate = lr;
    }
}
