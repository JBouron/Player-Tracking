#include <opencv2/core/core.hpp>
#include "../headers/bgsubstractor.h"
#include "../headers/debug.h"

namespace tmd {
    BGSubstractor::BGSubstractor(std::string input_video_path, cv::Mat
    static_mask,
                                 int camera_index, int
                                 starting_frame, int step_size) {
        m_bgs = new cv::BackgroundSubtractorMOG2(tmd::Config::bgs_history,
                                                 tmd::Config::bgs_threshold,
                                                 tmd::Config::bgs_detect_shadows);
        m_starting_frame = starting_frame;
        m_step_size = step_size;
        if (m_bgs == NULL) {
            throw std::bad_alloc();
        }

        if (static_mask.channels() != 1) {
            throw std::invalid_argument("The mask must only have 1 dimension : it's a binary image");
        }
        m_static_mask = static_mask.clone();

        m_learning_rate = tmd::Config::bgs_learning_rate;
        tmd::debug("BGSubstractor", "BGSubstractor", "bgs created.");

        m_input_video_path = input_video_path;
        m_input_video = new cv::VideoCapture(input_video_path);
        m_input_video->set(CV_CAP_PROP_POS_FRAMES, m_starting_frame);
        if (m_input_video == NULL || !m_input_video->isOpened()) {
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "input video is not valid (NULL or not opened).");
        }
        cv::VideoCapture cap(input_video_path);
        cv::Mat bg;
        cap.read(bg);
        cv::Mat mask;
        m_bgs->operator()(bg, mask, m_learning_rate);

        tmd::debug("BGSubstractor", "BGSubstractor", "valid input video.");
        m_camera_index = camera_index;
        if (!(0 <= m_camera_index && m_camera_index < 8)) {
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "invalid camera index " +
                                        m_camera_index);
        }

        tmd::debug("BGSubstractor", "BGSubstractor", "valid camera index");
        m_frame_index = m_starting_frame;
        m_total_frame_count = (m_input_video->get(CV_CAP_PROP_FRAME_COUNT));
        tmd::debug("BGSubstractor", "BGSubstractor", "m_total_frame_count = "
                                                     + std::to_string(m_total_frame_count));
        tmd::debug("BGSubstractor", "BGSubstractor", "exiting method");
    }

    BGSubstractor::~BGSubstractor() {
        // cv::Ptr free the bgs for us.
    }

    frame_t *BGSubstractor::next_frame() {
        frame_t *frame = new frame_t;
        bool frame_extracted = m_input_video->read(frame->original_frame);
        if (!frame_extracted) {
            tmd::debug("BGSubstractor", "next_frame", "No frame left, "
                                                              "returning NULL after " + std::to_string(m_frame_index) +
                                                      " frames");
            delete frame;
            return NULL;
        }
        frame->frame_index = m_frame_index;
        m_bgs->operator()(frame->original_frame,
                          frame->mask_frame,
                          m_learning_rate);
        frame->camera_index = m_camera_index;

        //applying static frame
        for (int row = 0; row < m_static_mask.rows; row++) {
            for (int col = 0; col < m_static_mask.cols; col++) {
                if (m_static_mask.at<uchar>(row, col) == 0) {
                    frame->mask_frame.at<uchar>(row, col) = 0;
                }
            }
        }

        //second pass
        cv::Mat mask_copy(m_static_mask.rows, m_static_mask.cols, CV_8U);
        cv::Mat checked_pixels;
        checked_pixels = cv::Mat::zeros(m_static_mask
                .rows, m_static_mask.cols, CV_8U);
        frame->mask_frame.copyTo(mask_copy);
        int buffer_size = 2;
        int count_threshold = 5;

        for (int row = 0; row < m_static_mask.rows; row++) {
            for (int col = 0; col < m_static_mask.cols; col++) {
                int current_value = frame->mask_frame.at<uchar>(row, col);
                if (current_value != 0) {
                    for (int neighbour_col = -buffer_size; neighbour_col <= buffer_size; neighbour_col++) {
                        for (int neighbour_row = -buffer_size; neighbour_row <= buffer_size; neighbour_row++) {
                            if (col + neighbour_col > 0 && col + neighbour_col < m_static_mask.cols
                                && row + neighbour_row > 0 && row + neighbour_row < m_static_mask.rows) {
                                if (checked_pixels.at<uchar>(row + neighbour_row, col + neighbour_col) == 0) {
                                    if (count_neighbours_in_fg(frame->mask_frame, col + neighbour_col,
                                                               row + neighbour_row, buffer_size) > count_threshold) {
                                        mask_copy.at<uchar>(row + neighbour_row, col + neighbour_col) = 255;
                                    }
                                    else {
                                        mask_copy.at<uchar>(row + neighbour_row, col + neighbour_col) = 0;
                                    }
                                    checked_pixels.at<uchar>(row + neighbour_row, col + neighbour_col) = 255;
                                }
                            }
                        }
                    }
                }
            }
        }
        frame->mask_frame.release();
        mask_copy.copyTo(frame->mask_frame);
        mask_copy.release();
        checked_pixels.release();
        m_frame_index += m_step_size;
        jump_to_frame(m_frame_index);
        return frame;
    }

    int BGSubstractor::count_neighbours_in_fg(cv::Mat frame, int x, int y, int buffer_size) {
        int count = 0;
        for (int row = -buffer_size; row <= buffer_size; row++) {
            for (int col = -buffer_size; col <= buffer_size; col++) {
                if (x + col > 0 && x + col < frame.cols && y + row > 0 && y + row < frame.rows) {
                    if (frame.at<uchar>(row + y, col + x) != 0) {
                        count++;
                    }
                }
            }
        }
        return count;
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

    void BGSubstractor::jump_to_frame(int index) {
        delete m_input_video;
        cv::VideoCapture *newCapture = new cv::VideoCapture(m_input_video_path);
        newCapture->set(CV_CAP_PROP_POS_FRAMES, index);
        m_input_video = newCapture;
        m_frame_index = index;
    }

    int BGSubstractor::get_current_frame_index() {
        return  (m_frame_index);
    }
}