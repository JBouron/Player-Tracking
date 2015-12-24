#include "../headers/multithreaded_pipeline.h"
#include "../headers/frame_t.h"

namespace tmd{
    MultithreadedPipeline::MultithreadedPipeline(std::string video_path,
                                                 int thread_count,
                                                 std::string model_file) :
    Pipeline(video_path, model_file){
        m_threads_ready = false;
        m_frame_pos = 0;
        m_thread_count = thread_count;
        if (m_thread_count <= 0){
            throw std::invalid_argument("Error : In nultithreaded pipeline : "
                                                "negative thread count");
        }

        m_pipeline_threads = new tmd::PipelineThread*[m_thread_count];
    }

    MultithreadedPipeline::~MultithreadedPipeline() {
        for (int i = 0 ; i < m_thread_count ; i ++){
            m_pipeline_threads[i]->request_stop();
            delete m_pipeline_threads[i];
        }
        delete[] m_pipeline_threads;
    }

    frame_t* MultithreadedPipeline::next_frame(){
        int thread_id = m_frame_pos % m_thread_count;
        tmd::frame_t* frame = m_pipeline_threads[thread_id]->pop_buffer();
        m_frame_pos += m_step;
        return frame;
    }

    void MultithreadedPipeline::set_frame_step_size(int step){
        if (m_running || m_threads_ready){
            throw std::runtime_error("Error : pipelines/threads are already "
                                             "running/ready");
        }
        else{
            m_step = step;
        }
    }

    void MultithreadedPipeline::set_start_frame(int frame_index){
        if (m_running || m_threads_ready){
            throw std::runtime_error("Error : pipelines/threads are already "
                                             "running/ready");
        }
        else{
            m_start = frame_index;
            m_video->set(CV_CAP_PROP_POS_FRAMES, frame_index);
            m_frame_pos = m_start;
        }
    }

    void MultithreadedPipeline::set_end_frame(int frame_index){
        if (m_running || m_threads_ready){
            throw std::runtime_error("Error : pipelines/threads are already "
                                             "running/ready");
        }
        else{
            m_end = frame_index;
        }
    }

    void MultithreadedPipeline::create_threads(){
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
            m_pipeline_threads[i] = new PipelineThread(threadId,
                                                       starting_frame,
                                                       ending_frame,
                                                       m_video_path, m_step);
        }
        m_threads_ready = true;
    }
}