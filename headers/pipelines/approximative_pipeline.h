#ifndef BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
#define BACHELOR_PROJECT_REAL_TIME_PIPELINE_H

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "multithreaded_pipeline.h"

namespace tmd{
    /*
     * Class holding a real time pipeline.
     */

    class ApproximativePipeline : public Pipeline{

    public:
        ApproximativePipeline(const std::string &video_folder, int camera_index,
                         int thread_count, int start_frame, int end_frame,
                         int box_step);

        ~ApproximativePipeline();

        frame_t* next_frame();

    private:
        void jump_video_to_next_frame();

        int m_box_step;
        int m_frame_pos;
        tmd::Pipeline *m_pipeline;
        tmd::frame_t *m_last_frame_computed;

        double m_frame_delay; // minimum time between to frames.
        double m_last_frame_time;
    };
}

#endif //BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
