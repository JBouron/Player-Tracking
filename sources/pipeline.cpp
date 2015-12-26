#include "../headers/pipeline.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/blob_separator.h"
#include "../headers/frame_t.h"
#include "../headers/player_t.h"


namespace tmd {
    Pipeline::Pipeline(std::string video_path, std::string model_file, unsigned
    char camera_index, int start_frame, int end_frame, int step_size) {
        m_video_path = video_path;

        m_video = new cv::VideoCapture;
        m_video->open(video_path); // TODO : Valkyrie reports a memleak here ...
        if (!m_video->isOpened()) {
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " pipeline.");
        }
        m_start = start_frame;
        m_step = step_size;
        m_end = end_frame;
        m_camera_index = camera_index;
    }

    Pipeline::~Pipeline() {
        m_video->release();
        delete m_video;
    }
}
