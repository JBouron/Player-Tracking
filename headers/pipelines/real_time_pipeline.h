#ifndef BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
#define BACHELOR_PROJECT_REAL_TIME_PIPELINE_H

#include "multithreaded_pipeline.h"

namespace tmd{
    /*
     * Class holding a real time pipeline.
     */

    class RealTimePipeline : public Pipeline{

    public:
        RealTimePipeline(const std::string &video_folder, int camera_index,
                         int thread_count, int start_frame, int end_frame,
                         int box_step);

        frame_t* next_frame();

    private:
        void jump_video_to_next_frame();

        int m_box_step;
        int m_frame_pos;
        tmd::Pipeline *m_pipeline;
        tmd::frame_t *m_last_frame_computed;

        double m_frame_delay; // minimum time between to frames.
    };
}

#endif //BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
