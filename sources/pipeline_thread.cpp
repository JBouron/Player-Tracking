#include "../headers/pipeline_thread.h"

namespace tmd{
    PipelineThread::PipelineThread(int thread_id, std::string video_path,
                                   int step_size) {
        m_id = thread_id;
        m_step_size = step_size;
        m_starting_frame = m_id;
        // TODO : Remove hard coded values using the config file.
        m_pipeline = new tmd::Pipeline(video_path, "./res/bgs_masks/mask_ace0"
                ".jpg", 0, "./res/xmls/person.xml", false, true,
           "./res/pipeline_results/complete_pipeline/uni/with blob separator/");
        m_pipeline->set_start_frame(m_id);
        m_pipeline->set_frame_step_size(m_step_size);
        m_worker = std::thread(&PipelineThread::extract_from_pipeline,
                               std::ref(this));
    }

    PipelineThread::~PipelineThread(){
        delete m_pipeline;
    }

    std::vector<tmd::player_t*> PipelineThread::pop_buffer(){
        std::vector<tmd::player_t*> head;
        bool done = false;
        while (!done) {
            {   // Trick here : having a scope inside the while method so
                // that the lock is released.
                std::lock_guard<std::mutex> lock(m_buffer_lock);
                if (m_buffer.size() > 0){
                    head = m_buffer[0];
                    m_buffer.pop_front();
                    done = true;
                }
                // If there is nothing in the buffer, we wait for the thread
                // to fill it.
            }
        }
        return head;
    }

    void PipelineThread::extract_from_pipeline(){
        std::vector<tmd::player_t*> next_buffer_entry =
                m_pipeline->next_players();
        this->push_buffer(next_buffer_entry);
    }

    void PipelineThread::push_buffer(std::vector<tmd::player_t*> players){
        std::lock_guard<std::mutex> lock(m_buffer_lock);
        tmd::debug("PipelineThread", "push_buffer", "Thread " +
                std::to_string(m_id) + " push entry in buffer");
        m_buffer.push_back(players);
    }
}