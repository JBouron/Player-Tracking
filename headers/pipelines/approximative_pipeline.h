#ifndef BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
#define BACHELOR_PROJECT_REAL_TIME_PIPELINE_H

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "multithreaded_pipeline.h"

namespace tmd{

    /**
     * The approximative pipeline allow the user to draw the boxes only at a
     * certain rate.
     * The performances are then better than with a normal Pipeline.
     * It allow us to reach Real-Time if the rate is correctly chosen.
     */
    class ApproximativePipeline : public Pipeline{

    public:
        /**
         * Constructor of the Approximative Pipeline.
         * video_folder : The folder containing the video.
         * camera_index : The camera index.
         * thread_count : The number of threads to use.
         * start_frame : The index of the first frame to begin.
         * end_frame : The index of the last frame to compute.
         * box_step : The number of frames before recomputing the boxes.
         */
        ApproximativePipeline(const std::string &video_folder, int camera_index,
                         int thread_count, int start_frame, int end_frame,
                         int box_step);

        /**
         * Destrucrtor of the Approximative Pipeline.
         */
        ~ApproximativePipeline();

        /**
         * Returns the next frame.
         *
         * /!\ BE CAREFUL /!\
         * As the frames attributes don't change when the boxes are not
         * refreshed (except for the image obviously), the frame_t* returned
         * is always the same. Thus the user MUST NOT free the frame after
         * using it, nor modify it.
         */
        frame_t* next_frame();

    private:
        int m_box_step;
        int m_frame_pos;
        tmd::Pipeline *m_pipeline;
        tmd::frame_t *m_last_frame_computed;

        double m_frame_delay; // Minimum time between to frames.
        double m_last_frame_time; // Last time a frame was returned.
    };
}

#endif //BACHELOR_PROJECT_REAL_TIME_PIPELINE_H
