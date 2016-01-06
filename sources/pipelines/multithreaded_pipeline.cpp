#include "../../headers/pipelines/multithreaded_pipeline.h"

namespace tmd {
    MultithreadedPipeline::MultithreadedPipeline(std::string video_folder,
             int camera_index, int thread_count, int start_frame, int end_frame,
             int step_size) : Pipeline(video_folder, camera_index, start_frame,
             end_frame, step_size) {

        m_frame_pos = m_start;
        m_thread_count = thread_count;
        if (m_thread_count <= 0) {
            throw std::invalid_argument("Error : In nultithreaded pipeline : "
                                                "negative thread count");
        }
        m_next_thread_to_use = 0;
        m_pipeline_threads = new tmd::PipelineThread *[m_thread_count];

        schedule_threads(video_folder);
    }

    MultithreadedPipeline::~MultithreadedPipeline() {
        for (int i = 0; i < m_thread_count; i++) {
            delete m_pipeline_threads[i];
        }
        delete[] m_pipeline_threads;
    }

    frame_t *MultithreadedPipeline::next_frame() {
        int thread_id = m_next_thread_to_use;
        tmd::frame_t *frame = m_pipeline_threads[thread_id]->pop_buffer();
        m_frame_pos += m_step;
        m_next_thread_to_use = (m_next_thread_to_use + 1) % m_thread_count;
        return frame;
    }

    void MultithreadedPipeline::schedule_threads(std::string video_folder) {
        tmd::debug("MultithreadedPipeline", "create_threads", "Creating "
                "threads");
        m_end = m_end - (m_end % m_step);
        int mod = (m_end / m_thread_count) % m_thread_count;

        for (int i = 0; i < m_thread_count; i++) {
            int threadId = i;
            int starting_frame = m_start + threadId * m_step;
            int ending_frame;
            if (threadId <= mod) {
                ending_frame = m_end - (mod - threadId) * m_step;
            }
            else {
                ending_frame = m_end - (mod - threadId) * m_step -
                               m_thread_count * m_step;
            }
            int step = m_step * m_thread_count;

            tmd::debug("MultithreadedPipeline", "create_threads", "Creating "
                          "thread " + std::to_string(i) +
                          " starting_frame = " +
                          std::to_string(starting_frame) + " ending_frame = " +
                          std::to_string(ending_frame) + " step = " +
                          std::to_string(step));

            m_pipeline_threads[i] = new PipelineThread(video_folder,
                                                       m_camera_index, threadId,
                                                       starting_frame,
                                                       ending_frame, step);
        }
    }
}