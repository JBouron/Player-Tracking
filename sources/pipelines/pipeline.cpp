#include "../../headers/pipelines/pipeline.h"

namespace tmd {
    Pipeline::Pipeline(std::string video_folder, int camera_index,
                       int start_frame, int end_frame, int step_size) {
        m_video_path = video_folder + "/ace_" + std::to_string(camera_index)
                       + ".mp4";

        m_video = new cv::VideoCapture;
        m_video->open(m_video_path);
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
