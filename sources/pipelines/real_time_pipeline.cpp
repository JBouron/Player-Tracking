#include "../../headers/pipelines/real_time_pipeline.h"
#include "../../headers/data_structures/frame_t.h"

namespace tmd{
    RealTimePipeline::RealTimePipeline(const std::string &video_folder, int
    thread_count, float box_refresh_rate, int camera_index,
                     int start_frame, int end_frame)
            : Pipeline(video_folder, camera_index, start_frame, end_frame,
                       1){
        m_video->set(CV_CAP_PROP_POS_FRAMES, start_frame);
        m_last_frame_computed = NULL;
        m_frame_pos = start_frame;
        double video_fps =  (m_video->get(CV_CAP_PROP_FPS));
        m_box_refresh_modulus = static_cast<int>(video_fps / box_refresh_rate);
        // TODO : Avoid using MultithreadedPipeline when thread_count = 1.
        m_pipeline = new tmd::MultithreadedPipeline(video_folder,
                                                    camera_index,
                                                    thread_count, start_frame,
                                            end_frame, m_box_refresh_modulus);
    }

    frame_t* RealTimePipeline::next_frame(){
        cv::Mat video_frame;
        if (!m_video->read(video_frame)){
            return NULL;
        }

        if ((m_frame_pos - m_start) % m_box_refresh_modulus == 0){
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
        return frame;
    }

    void RealTimePipeline::jump_video_to_next_frame(){
        cv::Mat dummy;
        for (int i = 0 ; i < m_step - 1 ; i ++){
            m_video->read(dummy);
        }
    }
}
