#ifndef BACHELOR_PROJECT_SIMPLE_PIPELINE_H
#define BACHELOR_PROJECT_SIMPLE_PIPELINE_H

#include "pipeline.h"

namespace tmd{
    /**
     * Class representing a simple pipeline, running on one thread only.
     */
    class SimplePipeline : public Pipeline{

    public:
        SimplePipeline(std::string video_path, std::string model_file);

        ~SimplePipeline();

        frame_t* next_frame();

        /**
         * Sets the properties of the bgs.
         */
        void set_bgs_properties(float threshold, int history_size, float
        learning_rate);

        /**
         * Set the step size between to consecutive extracted frames.
         */
        void set_frame_step_size(int step);

        /**
         * Set the starting frame index.
         * The extraction must not begun before this operation.
         */
        void set_start_frame(int frame_index);

        /**
         * Set the frame index
         */
        void set_end_frame(int frame_index);

    private:
        /**
         * Fetch the next frame from the BGS.
         */
        tmd::frame_t* fetch_next_frame();

        /**
         * Extract the players from the given frames, and set their property
         * (team, features, ...)
         */
        void extract_players_from_frame(tmd::frame_t* frame);

        tmd::BGSubstractor      *m_bgSubstractor;
        tmd::PlayerExtractor    *m_playerExtractor;
        tmd::FeaturesExtractor  *m_featuresExtractor;
        tmd::FeatureComparator  *m_featuresComparator;
    };
}

#endif //BACHELOR_PROJECT_SIMPLE_PIPELINE_H
