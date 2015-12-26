#include "../headers/multithreaded_pipeline.h"
#include "../headers/frame_t.h"

namespace tmd{
    MultithreadedPipeline::MultithreadedPipeline(std::string video_path,
                                                 int thread_count,
                                                 std::string model_file, int
                                                 start_frame, int end_frame, int step_size) :
    Pipeline(video_path, model_file, start_frame, end_frame, step_size){
        m_threads_ready = false;
        m_frame_pos = m_start;
        m_thread_count = thread_count;
        if (m_thread_count <= 0){
            throw std::invalid_argument("Error : In nultithreaded pipeline : "
                                                "negative thread count");
        }
        m_next_thread_to_use = 0;
        m_pipeline_threads = new tmd::PipelineThread*[m_thread_count];
        create_threads();
    }

    MultithreadedPipeline::~MultithreadedPipeline() {
        for (int i = 0 ; i < m_thread_count ; i ++){
            m_pipeline_threads[i]->request_stop();
            delete m_pipeline_threads[i];
        }
        delete[] m_pipeline_threads;
    }

    frame_t* MultithreadedPipeline::next_frame(){
        if (!m_threads_ready){
            create_threads();
        }
        int thread_id = m_next_thread_to_use;
        tmd::frame_t* frame = m_pipeline_threads[thread_id]->pop_buffer();
        m_frame_pos += m_step;
        m_next_thread_to_use = (m_next_thread_to_use + 1)%m_thread_count;
        return frame;
    }

    void MultithreadedPipeline::create_threads(){
        tmd::debug("MultithreadedPipeline", "create_threads", "Creating "
                "threads");
        int mod = m_end % m_thread_count;
        for (int i = 0 ; i < m_thread_count ; i ++){
            int threadId = i;
            int starting_frame = m_start + threadId*m_step;
            int ending_frame;
            if (threadId < mod){
                ending_frame = threadId - mod + m_end;
            }
            else{
                ending_frame = threadId - mod + m_end + m_thread_count;
            }
            int step = m_step * m_thread_count;
            tmd::debug("MultithreadedPipeline", "create_threads", "Creating "
                    "thread " + std::to_string(i) + " starting_frame = " +
                    std::to_string(starting_frame) + " ending_frame = " +
                    std::to_string(ending_frame) + " step = "  +
                    std::to_string(step));
            m_pipeline_threads[i] = new PipelineThread(threadId,
                                                       starting_frame,
                                                       ending_frame,
                                                       m_video_path, step);
        }
        m_threads_ready = true;
    }
}