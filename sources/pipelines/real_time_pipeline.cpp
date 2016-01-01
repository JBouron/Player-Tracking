#include "../../headers/pipelines/real_time_pipeline.h"
#include "../../headers/data_structures/frame_t.h"

namespace tmd{
    RealTimePipeline::RealTimePipeline(const std::string &video_folder,
           int camera_index, int thread_count, int start_frame, int end_frame,
                                       int box_step)
            : Pipeline(video_folder, camera_index, start_frame, end_frame,
                       1){
        m_video->set(CV_CAP_PROP_POS_FRAMES, start_frame);
        m_last_frame_computed = NULL;
        m_frame_pos = start_frame;
        m_box_step = box_step;

        double fps = m_video->get(CV_CAP_PROP_FPS);
        m_frame_delay = 1.0 / fps;

        if (thread_count == 1){
            m_pipeline = new tmd::SimplePipeline(video_folder, camera_index,
                                             start_frame, end_frame, box_step);
        }
        else{
            m_pipeline = new tmd::MultithreadedPipeline(video_folder,
                                                        camera_index,
                                                        thread_count, start_frame,
                                                end_frame, m_box_step);
        }
    }

    frame_t* RealTimePipeline::next_frame(){
        double time_start = cv::getTickCount();
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
        frame->original_frame = video_frame;
        frame->mask_frame = m_last_frame_computed->mask_frame;
        frame->blobs = m_last_frame_computed->blobs;
        frame->camera_index = m_last_frame_computed->camera_index;
        frame->colored_mask_frame = m_last_frame_computed->colored_mask_frame;
        frame->frame_index = m_last_frame_computed->frame_index;
        frame->players = m_last_frame_computed->players;
        /*size_t player_count = m_last_frame_computed->players.size();
        for (int i = 0 ; i < player_count ; i ++){
            frame->players.push_back(new tmd::player_t);
            memcpy((frame->players[i]), m_last_frame_computed->players[i],
                   sizeof(tmd::player_t));
        }*/
        //jump_video_to_next_frame();
        m_frame_pos += m_step;
        while ((cv::getTickCount() - time_start) / cv::getTickFrequency() <
                m_frame_delay);
        return frame;
    }

    void RealTimePipeline::jump_video_to_next_frame(){
        cv::Mat dummy;
        for (int i = 0 ; i < m_step - 1 ; i ++){
            m_video->read(dummy);
        }
    }
}
