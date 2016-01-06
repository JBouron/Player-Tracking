#include "../../headers/background_subtractor/bgsubstractor.h"

namespace tmd {
    BGSubstractor::BGSubstractor(std::string video_folder, int camera_index, int
    starting_frame, int ending_frame, int step_size) {
        m_bgs = new cv::BackgroundSubtractorMOG2(tmd::Config::bgs_history,
                                                 tmd::Config::bgs_threshold,
                                                 tmd::Config::bgs_detect_shadows);
        m_starting_frame = starting_frame;
        m_ending_frame = ending_frame;
        m_step_size = step_size;


        m_learning_rate = tmd::Config::bgs_learning_rate;
        tmd::debug("BGSubstractor", "BGSubstractor", "bgs created.");

        m_input_video_path = video_folder + "ace_" + std::to_string
                (camera_index) + ".mp4";

        std::string mask_path = tmd::Config::mask_folder + "mask_ace" +
                                std::to_string(camera_index) + ".jpg";
        m_static_mask = cv::imread(mask_path, 0);

        // Take the first frame of the video and take it as the background
        // model.
        m_input_video.open(m_input_video_path);
        if (!m_input_video.isOpened()) {
            throw std::invalid_argument("Error in BGSubstractor constructor, "
                                                "input video is not valid (NULL or not opened).");
        }

        cv::Mat bg;
        m_input_video.read(bg);
        cv::Mat mask;
        m_bgs->operator()(bg, mask, m_learning_rate);

        m_input_video.open(m_input_video_path);
        m_input_video.set(CV_CAP_PROP_POS_FRAMES, m_starting_frame);
        if (!m_input_video.isOpened()) {
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
        m_frame_index = m_starting_frame;
        m_total_frame_count = (m_input_video.get(CV_CAP_PROP_FRAME_COUNT));
        tmd::debug("BGSubstractor", "BGSubstractor", "m_total_frame_count = "
                                                     + std::to_string(m_total_frame_count));
        tmd::debug("BGSubstractor", "BGSubstractor", "exiting method");
    }

    BGSubstractor::~BGSubstractor() {
        // cv::Ptr free the bgs for us.
        m_input_video.release();
    }

    frame_t *BGSubstractor::next_frame() {
        frame_t *frame = new frame_t;
        bool frame_extracted = m_input_video.read(frame->original_frame);
        if (!frame_extracted || m_frame_index > m_ending_frame) {
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

        //second pass : "Reduce" the resolution of the mask image.
        cv::Mat mask_copy(m_static_mask.rows, m_static_mask.cols, CV_8U);
        cv::Mat checked_pixels;
        checked_pixels = cv::Mat::zeros(m_static_mask
                                                .rows, m_static_mask.cols, CV_8U);
        frame->mask_frame.copyTo(mask_copy);
        int buffer_size = tmd::Config::bgs_blob_buffer_size;
        int count_threshold = tmd::Config::bgs_blob_threshold_count;

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
        cv::Mat coloredMask = get_colored_mask_for_frame(frame);
        frame->colored_mask_frame = coloredMask;
        step();
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
        m_input_video.set(CV_CAP_PROP_POS_FRAMES, index);
        m_frame_index = index;
    }

    int BGSubstractor::get_current_frame_index() {
        return (m_frame_index);
    }

    void BGSubstractor::step(){
        cv::Mat dummy;
        for (int i = 0 ; i < m_step_size - 1 ; i ++){
            m_input_video.read(dummy);
        }
        m_frame_index += m_step_size;
    }
}