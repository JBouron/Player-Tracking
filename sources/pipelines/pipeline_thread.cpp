#include "../../headers/pipelines/pipeline_thread.h"

namespace tmd {
    PipelineThread::PipelineThread(std::string video_folder, int camera_index,
                                   int thread_id, int starting_frame,
                                   int ending_frame, int step_size) {
        m_id = thread_id;
        m_step_size = step_size;
        m_starting_frame = starting_frame;
        m_frame_idx = m_starting_frame;
        m_ending_frame = ending_frame;
        m_pipeline = new tmd::SimplePipeline(video_folder, camera_index,
                                     starting_frame, ending_frame, step_size);
        m_stop_request = false;

        m_done = false;
        m_worker = std::thread(&PipelineThread::extract_from_pipeline,
                               std::ref(*this));
    }

    PipelineThread::~PipelineThread() {
        if (!m_done) {
            m_stop_request = true;
            while (!m_worker_stopped) {
                tmd::debug("PipelineThread", "~PipelineThread", "Waiting for "
                        "thread to stop.");
            }
        }
        delete m_pipeline;
    }

    tmd::frame_t *PipelineThread::pop_buffer() {
        if (m_done){
            return NULL;
        }
        tmd::frame_t *head = NULL;
        while (true) {
            {
                std::lock_guard<std::mutex> lock(m_buffer_lock);
                if (m_buffer.size() > 0) {
                    head = m_buffer[0];
                    m_buffer.pop_front();
                    m_done = (head == NULL);
                    return head;
                }
            }
        }
    }

    void PipelineThread::extract_from_pipeline() {
        while (!m_stop_request && m_frame_idx <= m_ending_frame) {
            tmd::debug("PipelineThread", "extract_from_pipeline", "Thread " +
                              std::to_string(m_id) + " calling next_players()");
            tmd::frame_t *next_buffer_entry = m_pipeline->next_frame();
            if (next_buffer_entry == NULL) {
                break;
            }
            tmd::debug("PipelineThread", "extract_from_pipeline", "Thread " +
                                              std::to_string(m_id) + " : Done");
            this->push_buffer(next_buffer_entry);
            m_frame_idx += m_step_size;
        }
        this->push_buffer(NULL); // Indicating the end.
        std::lock_guard<std::mutex> lock(m_buffer_lock);
        m_worker_stopped = true;
    }

    void PipelineThread::push_buffer(tmd::frame_t *frame) {
        std::lock_guard<std::mutex> lock(m_buffer_lock);
        tmd::debug("PipelineThread", "push_buffer", "Thread " +
                                std::to_string(m_id) + " push entry in buffer");
        m_buffer.push_back(frame);
    }
}