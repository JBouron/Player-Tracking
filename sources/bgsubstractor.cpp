#include <opencv2/core/core.hpp>
#include "../headers/bgsubstractor.h"
#include "../headers/debug.h"
#include "../headers/frame_t.h"

namespace tmd {
    BGSubstractor::BGSubstractor(cv::VideoCapture *input_video, cv::Mat static_mask,
                                 unsigned char camera_index, float threshold,
                                 int history, float learning_rate) {
        m_bgs = new cv::BackgroundSubtractorMOG2(history, threshold,
                                                 TMD_BGS_DETECTS_SHADOWS);
        m_start = true;

        if (m_bgs == NULL) {
            throw std::bad_alloc();
        }

        if(static_mask.channels() != 1){
            throw std::invalid_argument("The mask must only have 1 dimension : it's a binary image");
        }
        m_static_mask = static_mask.clone();

        m_learning_rate = learning_rate;
        tmd::debug("BGSubstractor", "BGSubstractor", "bgs created.");

        m_input_video = input_video;
        if (m_input_video == NULL || !m_input_video->isOpened()) {
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                            "input video is not valid (NULL or not opened).");
        }

        tmd::debug("BGSubstractor", "BGSubstractor", "valid input video.");
        m_camera_index = camera_index;
        if (!(0 <= m_camera_index && m_camera_index < 8)) {
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "invalid camera index " +
                                        m_camera_index);
        }

        tmd::debug("BGSubstractor", "BGSubstractor", "valid camera index");
        m_frame_index = 0;
        m_total_frame_count =  (m_input_video->get(CV_CAP_PROP_FRAME_COUNT));
        tmd::debug("BGSubstractor", "BGSubstractor", "m_total_frame_count = "
                                     + std::to_string(m_total_frame_count));
        tmd::debug("BGSubstractor", "BGSubstractor", "exiting method");
    }

    BGSubstractor::~BGSubstractor(){
        // cv::Ptr free the bgs for us.
    }

    frame_t *BGSubstractor::next_frame() {
        frame_t *frame = new frame_t;
        bool frame_extracted = m_input_video->read(frame->original_frame);
        if (!frame_extracted){
            tmd::debug("BGSubstractor", "next_frame", "No frame left, "
                    "returning NULL after " + std::to_string(m_frame_index) +
                    " frames");
            delete frame;
            return NULL;
        }
        frame->frame_index = m_frame_index;
        cv::Mat mask;

        if(m_start){
            frame->original_frame = cv::imread("./res/emptyroom.jpg");
            m_start = false;
        }
        m_bgs->operator()(frame->original_frame,
                                        frame->mask_frame,
                          m_learning_rate);
        frame->camera_index = m_camera_index;

        for(int row = 0; row < m_static_mask.rows; row++){
            for(int col = 0; col < m_static_mask.cols ; col++) {
                if(m_static_mask.at<uchar>(row, col) == 0){
                    frame->mask_frame.at<uchar>(row, col) = 0;
                }
            }
        }

        m_frame_index++;
        return frame;
    }

    void BGSubstractor::set_threshold_value(float th) {
        m_bgs->set("varThreshold", th);
    }

    void BGSubstractor::set_history_size(int s) {
        m_bgs->set("history", s);
    }

    void BGSubstractor::set_learning_rate(float lr) {
        m_learning_rate = lr;
    }

    bool BGSubstractor::jump_to_frame(int index) {
        m_input_video->set(CV_CAP_PROP_POS_FRAMES,static_cast<double>(index));
        m_frame_index = index;
    }

    int BGSubstractor::get_current_frame_index(){
        return static_cast<int> (m_frame_index);
    }
}
