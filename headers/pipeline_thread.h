#ifndef BACHELOR_PROJECT_PIPELINE_THREAD_H
#define BACHELOR_PROJECT_PIPELINE_THREAD_H

#include <string>
#include <deque>
#include <vector>
#include <mutex>
#include <thread>
#include "player_t.h"
#include "pipeline.h"
#include "debug.h"

namespace tmd{
    /**
     * Classe representing a thread on a pipeline.
     * It allow us to have a fine-grained multithreading on a pipeline.
     */
    class PipelineThread{
    public:
        /**
         * Constructor of the PipelineThread.
         * thread_id : the id of the thread, ie it's starting frame.
         * video_path : The video to operate on.
         * step_size : Frame count to jump over between 2 consecutive buffer
         * entries.
         */
        PipelineThread(int thread_id, std::string video_path, int step_size);

        /**
         * Destructor of the PipelineThread.
         */
        ~PipelineThread();

        /**
         * Get the top of the buffer (ie the oldest entry).
         * /!\ Can lead to waiting time due to dependencies between the
         * caller and *this.
         */
        std::vector<tmd::player_t*> pop_buffer();

    private:
        /**
         * Method executed by the working thread.
         */
        void extract_from_pipeline();
        /**
         * Add data to the end of the buffer.
         * Only used by the working thread.
         */
        void push_buffer(std::vector<tmd::player_t*> players);

        tmd::Pipeline *m_pipeline;
        std::thread m_worker;
        std::deque<std::vector<tmd::player_t*> > m_buffer;
        std::mutex m_buffer_lock;
        int m_starting_frame;
        int m_step_size;
        int m_id;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_THREAD_H
