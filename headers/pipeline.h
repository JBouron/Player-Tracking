#ifndef BACHELOR_PROJECT_PIPELINE_H
#define BACHELOR_PROJECT_PIPELINE_H

#include <string>
#include "frame_t.h"
#include "bgsubstractor.h"
#include "player_extractor.h"
#include "features_extractor.h"
#include "feature_comparator.h"
#include "write_buffer.h"

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
        Pipeline(std::string video_path, std::string static_mask_path, unsigned char camera_index,
                 std::string model_file, bool dpm
        = false, bool save_frames = false,
                 bool save_mask = false, std::string output_folder = "");

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
        frame_t* next_frame();


        /**
         * Sets the properties of the bgs.
         */
        void set_bgs_properties(float threshold, int history_size, float
            learning_rate);

        /**
         * Sets the properties of the dpm (if used for player extraction).
         */
        void set_dpm_properties(float overlapping_threshold, float
        score_threshold);

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

        static cv::Mat get_colored_mask_for_frame(frame_t* frame);

    private:
        cv::VideoCapture *m_video;
        tmd::BGSubstractor *m_bgSubstractor;
        tmd::PlayerExtractor *m_playerExtractor;
        tmd::FeaturesExtractor *m_featuresExtractor;
        tmd::FeatureComparator *m_featuresComparator;
        tmd::WriteBuffer *m_write_buffer;
        std::string m_video_path;
        unsigned char m_camera_index;
        std::string m_output_folder;
        bool m_save;
        bool m_running;
        bool m_using_dpm;
        int m_step;
        int m_start;
        int m_end;
    };
}

#endif //BACHELOR_PROJECT_PIPELINE_H
