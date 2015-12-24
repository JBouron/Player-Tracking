#include "../headers/pipeline_visualizer.h"

namespace tmd{
    PipelineVisualizer::PipelineVisualizer(tmd::Pipeline *pipeline,
                                       std::string video_path, int frame_rate){
        if (pipeline == NULL){
            throw std::invalid_argument("Error, passing a NULL pointer to the"
                                                " Pipeline Visualizer.");
        }
        m_video = new cv::VideoCapture(video_path);
        m_frame_step = frame_rate;
        m_pipeline = pipeline;
        m_frame_pos = 0;
        m_paused = false;
        m_window = SDLBinds::create_sdl_window
                (TMD_PIPELINE_VISUALIZER_WINDOW_NAME,
                 static_cast<int>(m_video->get(CV_CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(m_video->get(CV_CAP_PROP_FRAME_HEIGHT)));
    }

    PipelineVisualizer::~PipelineVisualizer(){
        delete m_video;
        SDLBinds::destroy_sdl_window(m_window);
    }

    void PipelineVisualizer::run() {
        int keyboard = 0;
        //fetch_next_players();
        while (keyboard != 27){
            if (m_frame_pos % m_frame_step == 0){
                // Wait the result for those frames.
                // Get the results for the next frames.
                for (int i = 0 ; i < m_players.size() ; i ++){
                    delete m_players[i];
                }
                m_players = m_next_players;
                fetch_next_players();
                m_pipeline_thread.join();
            }

            if (!m_paused){
                draw_next_frame();
                m_frame_pos += 1;
            }
            keyboard = cv::waitKey(100);
            if (keyboard == 32) {
                m_paused = !m_paused;
            }
        }
    }

    void PipelineVisualizer::draw_next_frame(){
        cv::Mat frame;
        m_video->read(frame);
        if (frame.empty()){
            return;
        }

        CvScalar torso_color;
        torso_color.val[0] = 255;
        torso_color.val[1] = 255;
        torso_color.val[2] = 0;
        torso_color.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;

        for (size_t i = 0 ; i < m_players.size() ; i ++){
            player_t *p = m_players[i];
            cv::rectangle(frame, m_players[i]->pos_frame,
                          get_team_color(m_players[i]->team), thickness,
                          line_type, shift);

            //show_body_parts(frame->original_frame, p);
            cv::Rect torso;
            torso.x = p->pos_frame.x + p->features.torso_pos.x;
            torso.y = p->pos_frame.y + p->features.torso_pos.y;
            torso.width = p->features.torso_pos.width;
            torso.height = p->features.torso_pos.height;
            cv::rectangle(frame, torso,
                          torso_color, thickness,
                          line_type, shift);
        }
        std::string frame_idx = std::to_string(m_frame_pos);
        cv::imwrite("./res/pipeline_visualizer/frame" + frame_idx + ".jpg",
                    frame);
        SDLBinds::imshow(m_window, frame);
    }

    void _fetch_players(tmd::Pipeline *pipeline, std::vector<tmd::player_t*>
    &next_players){
        next_players.clear();
        next_players = pipeline->next_players();
    }

    void PipelineVisualizer::fetch_next_players(){
        m_pipeline_thread = std::thread(_fetch_players, m_pipeline, std::ref
                (m_next_players));
    }
}