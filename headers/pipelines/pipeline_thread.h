#ifndef BACHELOR_PROJECT_PIPELINE_THREAD_H
#define BACHELOR_PROJECT_PIPELINE_THREAD_H

#include <string>
#include <deque>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <boost/lockfree/queue.hpp>
#include "../data_structures/player_t.h"
#include "pipeline.h"
#include "../misc/debug.h"
#include "simple_pipeline.h"

namespace tmd{
    /**
     * Class representing a thread running  a pipeline.
     * This thread is completely independent form the main thread so that it
     * only computes the frames and put them in a buffer.
     */
    class PipelineThread{
    public:
        /**
         * Constructor of the PipelineThread.
         * video_folder : Folder containing the video.
         * camera_index : The camera index.
         * thread_id : The id of this thread.
         * start_frame : The index of the first frame to begin.
         * end_frame : The index of the last frame to compute.
         * step_size : The "distance" between to consecutive frames.
         */
        PipelineThread(std::string video_folder, int camera_index, int thread_id
                , int starting_frame, int ending_frame, int step_size);

        /**
         * Destructor of the PipelineThread.
         */
        ~PipelineThread();

        /**
         * Get the top of the buffer (ie the oldest entry).
         * /!\ Can lead to waiting time due to dependencies between the
         * caller and *this.
         */
        tmd::frame_t* pop_buffer();

    private:
        /**
         * Method executed by the working thread.
         */
        void extract_from_pipeline();
        /**
         * Add data to the end of the buffer.
         * Only used by the working thread.
         */
        void push_buffer(tmd::frame_t* frame);

        tmd::Pipeline *m_pipeline;
        std::thread m_worker;
        //std::deque<tmd::frame_t* > m_buffer;
        boost::lockfree::queue<tmd::frame_t*> m_buffer;
        std::mutex m_buffer_lock;

        int m_starting_frame;
        int m_ending_frame;
        int m_frame_idx;
        int m_step_size;
        int m_id;
        bool m_done;

        std::atomic<bool> m_stop_request;
        std::atomic<bool> m_worker_stopped;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_THREAD_H
