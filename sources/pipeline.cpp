#include "../headers/pipeline.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/blob_separator.h"
#include "../headers/frame_t.h"
#include "../headers/player_t.h"


namespace tmd {
    Pipeline::Pipeline(std::string video_path, std::string model_file,
                       bool save_frames, std::string output_folder) {
        m_video_path = video_path;

        m_video = new cv::VideoCapture;
        m_video->open(video_path); // TODO : Valkyrie reports a memleak here ...
        if (!m_video->isOpened()) {
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " pipeline.");
        }

        m_running = false;

        m_start = 0;
        m_step = 1;
        m_end = -1;

        m_save = save_frames;
        m_output_folder = output_folder;
    }

    Pipeline::~Pipeline() {
        m_video->release();
        delete m_video;
    }

    cv::Mat Pipeline::get_colored_mask_for_frame(frame_t *frame) {
        cv::Mat resulting_image;
        frame->original_frame.copyTo(resulting_image);
        cv::Vec3b black;
        black.val[0] = 0;
        black.val[1] = 0;
        black.val[2] = 0;
        for (int c = 0; c < frame->mask_frame.cols; c++) {
            for (int r = 0; r < frame->mask_frame.rows; r++) {
                if (frame->mask_frame.at<uchar>(r, c) < 127) {
                    resulting_image.at<cv::Vec3b>(r, c) = black;
                }
            }
        }
        return resulting_image;
    }
}
