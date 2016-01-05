#ifndef BACHELOR_PROJECT_SIMPLE_PIPELINE_H
#define BACHELOR_PROJECT_SIMPLE_PIPELINE_H

#include "pipeline.h"
#include "../players_extraction/blob_based_extraction/blob_separator.h"

namespace tmd{
    /**
     * Class representing a simple pipeline, running on one thread only.
     */
    class SimplePipeline : public Pipeline{

    public:
        /**
         * Constructor of the Simple Pipeline
         * video_folder : Folder containing the video.
         * camera_index : The camera index.
         * start_frame : The index of the first frame to begin.
         * end_frame : The index of the last frame to compute.
         * step_size : The "distance" between to consecutive frames.
         */
        SimplePipeline(std::string video_folder, int camera_index,
                       int start_frame, int end_frame, int step_size);

        /**
         * Destructor of the Simple pipeline.
         */
        ~SimplePipeline();

        /**
         * Returns the next frame after computing it.
         */
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
