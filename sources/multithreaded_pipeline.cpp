#include "../headers/multithreaded_pipeline.h"

namespace tmd{
    MultithreadedPipeline::MultithreadedPipeline(std::string video_path,
                                                 int thread_count,
                                                 std::string model_file,
                                                 bool save_frames,
                                                 std::string output_folder) :
    Pipeline(video_path, model_file, save_frames, output_folder){
        m_threads_ready = false;
        m_thread_count = thread_count;
        if (m_thread_count <= 0){
            throw std::invalid_argument("Error : In nultithreaded pipeline : "
                                                "negative thread count");
        }

        m_pipeline_threads = new tmd::PipelineThread*[m_thread_count];
    }
}