#ifndef BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
#define BACHELOR_PROJECT_REAL_TIME_PIPELINE_H

#include "multithreaded_pipeline.h"

namespace tmd{
    /*
     * Class holding a real time pipeline.
     */

    class RealTimePipeline : public Pipeline{

    public:
        RealTimePipeline(const std::string &video_folder, int thread_count,
                         int box_refresh_rate, int camera_index,
                         int start_frame, int end_frame);

        frame_t* next_frame();

    private:
        void jump_video_to_next_frame();

        int m_box_refresh_modulus;
        int m_frame_pos;
        tmd::Pipeline *m_pipeline;
        tmd::frame_t *m_last_frame_computed;
    };
}

#endif //BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
