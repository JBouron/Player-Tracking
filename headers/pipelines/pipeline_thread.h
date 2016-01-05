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
     * Classe representing a thread on a pipeline.
     * It allow us to have a fine-grained multithreading on a pipeline.
     */
    class PipelineThread{
    public:
        /**
         * Constructor of the PipelineThread.
         * thread_id : the id of the thread.
         * global_starting_frame : The starting frame.
         * global_ending_frame : The ending frame.
         * video_path : The video to operate on.
         * step_size : Frame count to jump over between 2 consecutive buffer
         * entries.
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
