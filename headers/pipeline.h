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
         * _ dpm : Enable using dpm player extractor, if not enabled, the
         * blob-detection based player extractor will be used instead.
         * _ model_file : Path to the file containing the model of the person.
         * _ cluster_centers_file : Path to the file containing the cluster
         * centers.
         * _ save_frames : True to enable saving the frames into the output
         * folder.
         * _ output_folder : Path to the folder which will contain all the
         * saved frames (if save_frame is enabled).
         */
         Pipeline(std::string video_path,
                 std::string model_file, bool save_frames = false,
                 std::string output_folder = "");

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


        /**
         * Extract the next frame from the input video.
         * Run the full pipeline on it.
         * Then returns a vector containing all the player for this frame.
         * Along with their team, features, ...
         */
        virtual std::vector<tmd::player_t*> next_players() = 0;

        /**
         * Set the step size between to consecutive extracted frames.
         */
        virtual void set_frame_step_size(int step) = 0;

        /**
         * Set the starting frame index.
         * The extraction must not begun before this operation.
         */
        virtual void set_start_frame(int frame_index) = 0;

        /**
         * Set the frame index
         */
        virtual void set_end_frame(int frame_index) = 0;

        /**
         * Create a 'colored mask' ie all pixel belonging to the foreground
         * are in color whereas pixels from the background are black.
         */
        static cv::Mat get_colored_mask_for_frame(tmd::frame_t* frame);

    protected:

        cv::VideoCapture *m_video;

        std::string m_video_path;
        unsigned char m_camera_index;
        std::string m_output_folder;
        bool m_save;
        bool m_running;
        int m_step;
        int m_start;
        int m_end;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_H
