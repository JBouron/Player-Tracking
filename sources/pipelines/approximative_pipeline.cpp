#include "../../headers/pipelines/approximative_pipeline.h"
#include "../../headers/data_structures/frame_t.h"

namespace tmd{
    ApproximativePipeline::ApproximativePipeline(const std::string &video_folder
            ,int camera_index, int thread_count, int start_frame, int end_frame,
             int box_step) : Pipeline(video_folder, camera_index, start_frame,
                                      end_frame, 1){

        m_video->set(CV_CAP_PROP_POS_FRAMES, start_frame);
        m_last_frame_computed = NULL;
        m_frame_pos = start_frame;
        m_box_step = box_step;

        double fps = m_video->get(CV_CAP_PROP_FPS);
        m_frame_delay = 1.0 / fps;

        m_pipeline = new tmd::MultithreadedPipeline(video_folder, camera_index,
                                                    thread_count, start_frame,
                                                    end_frame, m_box_step);
    }

    ApproximativePipeline::~ApproximativePipeline(){
        delete m_pipeline;
        free_frame(m_last_frame_computed);
    }

    frame_t* ApproximativePipeline::next_frame(){
        cv::Mat video_frame;
        if (!m_video->read(video_frame)){
            return NULL;
        }
        if ((m_frame_pos - m_start) % m_box_step == 0){
            free_frame(m_last_frame_computed);
            m_last_frame_computed = m_pipeline->next_frame();
            if (m_last_frame_computed == NULL){
                return NULL;
            }
        }

        tmd::frame_t* frame = new frame_t;
        m_last_frame_computed->original_frame = video_frame;
        m_frame_pos += m_step;

        while ((cv::getTickCount() - m_last_frame_time) /
                        cv::getTickFrequency() <
                m_frame_delay);
        m_last_frame_time = cv::getTickCount();

        return m_last_frame_computed;
    }
}
