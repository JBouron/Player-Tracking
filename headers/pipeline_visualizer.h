#ifndef BACHELOR_PROJECT_PIPELINE_VISUALIZER_H
#define BACHELOR_PROJECT_PIPELINE_VISUALIZER_H

#include <thread>
#include "pipeline.h"

#define TMD_PIPELINE_VISUALIZER_WINDOW_NAME "Pipeline Visualizer"

namespace tmd{
    /**
     * Class representing a visualizer for the pipeline results.
     * The user can interact by pausing and zooming into the frames.
     */
    class PipelineVisualizer{
    public:
        /**
         * Constructor of the visualizer.
         * Takes a reference to the pipeline to be visualized.
         */
        PipelineVisualizer(tmd::Pipeline *pipeline, std::string video_path,
                           int frame_step);

        ~PipelineVisualizer();

        /**
         * Launch the visualizer.
         */
        void run();

    private:
        cv::Mat draw_next_frame();

        void fetch_next_players();

        cv::VideoCapture *m_video;
        std::thread m_pipeline_thread;
        tmd::Pipeline *m_pipeline;
        int m_frame_step;
        int m_frame_pos;
        bool m_paused;
        std::vector<tmd::player_t*> m_players;
        std::vector<tmd::player_t*> m_next_players;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_VISUALIZER_H
