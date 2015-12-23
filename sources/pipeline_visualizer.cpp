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
        cv::namedWindow(TMD_PIPELINE_VISUALIZER_WINDOW_NAME);
    }

    PipelineVisualizer::~PipelineVisualizer(){
        delete m_video;
    }

    void PipelineVisualizer::run() {
        int keyboard = 0;
        fetch_next_players();
        while (keyboard != 27){
            if (m_frame_pos % m_frame_step == 0){
                // Wait the result for those frames.
                m_pipeline_thread.join();
                // Get the results for the next frames.
                for (int i = 0 ; i < m_players.size() ; i ++){
                    delete m_players[i];
                }
                m_players = m_next_players;
                fetch_next_players();
            }
            else{
                if (!m_paused){
                    draw_next_frame();
                }
                keyboard = cv::waitKey(1);
                if (keyboard == 32){
                    m_paused = ! m_paused;
                }
            }
        }
    }

    cv::Mat PipelineVisualizer::draw_next_frame(){
        cv::Mat frame;
        m_video->read(frame);
        if (frame.empty()){
            return frame;
        }

        CvScalar color;
        color.val[0] = 255;
        color.val[1] = 0;
        color.val[2] = 255;
        color.val[3] = 255;
        CvScalar torso;
        torso.val[0] = 255;
        torso.val[1] = 255;
        torso.val[2] = 0;
        torso.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;

        for (size_t i = 0 ; i < m_players.size() ; i ++){
            tmd::player_t* p = m_players[i];
            std::vector<cv::Rect> parts = p->features.body_parts;
            for (int i = 0; i < parts.size(); i++) {
                CvRect r;
                r.x = parts[i].x + p->pos_frame.x;
                r.y = parts[i].y + p->pos_frame.y;
                r.width = parts[i].width;
                r.height = parts[i].height;
                cv::rectangle(frame, r, color, thickness, line_type, shift);
            }
        }
        return frame;
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