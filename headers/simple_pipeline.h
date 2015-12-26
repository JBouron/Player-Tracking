#ifndef BACHELOR_PROJECT_SIMPLE_PIPELINE_H
#define BACHELOR_PROJECT_SIMPLE_PIPELINE_H

#include "pipeline.h"

namespace tmd{
    /**
     * Class representing a simple pipeline, running on one thread only.
     */
    class SimplePipeline : public Pipeline{

    public:
        SimplePipeline(std::string video_folder, int camera_index,
                       int start_frame, int end_frame, int step_size);

        ~SimplePipeline();

        frame_t* next_frame();

        /**
         * Sets the properties of the bgs.
         */
        void set_bgs_properties(float threshold, int history_size, float
        learning_rate);

    private:
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
