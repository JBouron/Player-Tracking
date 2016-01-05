#ifndef BACHELOR_PROJECT_PIPELINE_H
#define BACHELOR_PROJECT_PIPELINE_H

#include <string>
#include "../players_extraction/blob_based_extraction/blob_player_extractor.h"
#include "../data_structures/frame_t.h"
#include "../background_subtractor/bgsubstractor.h"
#include "../players_extraction/player_extractor.h"
#include "../features_extraction/features_extractor.h"
#include "../features_comparison/feature_comparator.h"

namespace tmd{

    /**
     * The most important class. Link everything together. Take a frame
     * from the input video, apply the whole algorithm on it and then returns
     * the frames with its players, blobs, etc ...
     *
     * This class is abstract so that we can define multiple different
     * pipelines.
     */
    class Pipeline {
    public:
        /**
         * Constructor of the pipeline:
         * video_folder : Folder containing the video.
         * camera_index : The camera index.
         * start_frame : The index of the first frame to begin.
         * end_frame : The index of the last frame to compute.
         * step_size : The "distance" between to consecutive frames.
         */
         Pipeline(std::string video_folder, int camera_index, int start_frame,
                  int end_frame, int step_size);

        ~Pipeline();

        /**
         * Extract the next frame from the input video.
         * Run the full pipeline on this frame.
         * And finally return the result.
         *
         * If there is no frame left the method simply returns NULL.
         *
         * Note that the user has to take care of freeing the frames.
         */
        virtual frame_t* next_frame() = 0;

    protected:

        cv::VideoCapture *m_video;

        std::string m_video_path;
        int m_camera_index;
        int m_step;
        int m_start;
        int m_end;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_H
