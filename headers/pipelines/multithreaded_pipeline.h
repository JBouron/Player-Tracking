#include "pipeline.h"
#include "pipeline_thread.h"

#ifndef BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H
#define BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H

namespace tmd{
    /**
     * Class reprensenting a multithreaded pipeline.
     * The threads are scheduled in a fine-grained way :
     * Frame index : s s+1 s+2 s+3 ... e
     * Thread :      0  1   2   3  ... x
     */
    class MultithreadedPipeline : public Pipeline{

    public:
        /**
         * Constructor of the Multithreaded Pipeline.
         * video_folder : Folder containing the video.
         * camera_index : The camera index.
         * thread_count : The number of threads to use.
         * start_frame : The index of the first frame to begin.
         * end_frame : The index of the last frame to compute.
         * step_size : The "distance" between to consecutive frames.
         */
        MultithreadedPipeline(std::string video_folder, int camera_index, int
                thread_count, int start_frame, int end_frame, int step_size);

        ~MultithreadedPipeline();

        /**
         * Extract the next frame from the input video.
         * Run the full pipeline on this frame.
         * And finally return the result.
         *
         * If there is no frame left the method simply returns NULL.
         *
         * Note that the user has to take care of freeing the frames.
         */
        frame_t* next_frame();

    private:
        void schedule_threads(std::string video_folder);

        tmd::PipelineThread** m_pipeline_threads; // The threads.
        int m_thread_count;
        int m_frame_pos; // Current frame index.
        int m_next_thread_to_use; // Keep track of the next thread we should
                                    // use to get the next frame.
    };
}

#endif //BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H
