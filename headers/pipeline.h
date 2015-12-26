#ifndef BACHELOR_PROJECT_PIPELINE_H
#define BACHELOR_PROJECT_PIPELINE_H

#include <string>
#include "frame_t.h"
#include "bgsubstractor.h"
#include "player_extractor.h"
#include "features_extractor.h"
#include "feature_comparator.h"

namespace tmd{

    /**
     * The most important class. Link everything together. Take a frame in
     * and output the  same frame with color boxes.
     */
    class Pipeline {
    public:
        /**
         * Constructor of the pipeline:
         * _ video_path : the path to the video on which we will run the
         * pipeline
         * _ static_mask_path : the path to the static mask used for the
         * background subtraction
         * _ camera_index : The index of the camera.
         * _ model_file : Path to the file containing the model of the person.
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
