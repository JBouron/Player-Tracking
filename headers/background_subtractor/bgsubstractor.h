#ifndef TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H

#include <exception>
#include <stdexcept>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "../misc/debug.h"
#include "../data_structures/frame_t.h"
#include "../misc/config.h"

namespace tmd {

    /**
     * Wrapper of the background subtractor class from openCV.
     * This class take an input video and the user can retrieve each frame by
     * calling the next_frame method.
     * This can be seen as an iterator over the video, performing the
     * background subtraction over the returned frames.
     */
    class BGSubstractor {
    public:
        /**
         * Constructor of the Background Substractor.
         * input_video : The video to operate on.
         * camera_index : The index of the camera.
         * starting_frame : The index of the first frame.
         * ending_frame : The index of the last frame.
         * step_size : The step size between to frame returned by next_frame().
         */
        BGSubstractor(std::string video_folder, int camera_index,
                      int starting_frame = 0, int ending_frame = -1, int
                      step_size = 1);

        /**
         * Destructor of the BGS.
         */
        ~BGSubstractor();

        /**
         * Extract the next image from the input_video, apply BGS on it and
         * return a frame_t* containing the original image, the background
         * mask and the colored mask.
         *
         * Return NULL if there is no frame left in the input stream or if
         * the ending_frame has been reached.
         */
        frame_t *next_frame();

        /**
         * Set the bgs to a given frame.
         */
        void jump_to_frame(int index);

        /**
         * Set the color distance to use when extracting the background.
         */
        void set_threshold_value(float th);

        /**
         * Set the size of the history.
         */
        void set_history_size(int s);

        /**
         * Set the learning rate to use.
         */
        void set_learning_rate(float lr);

        /**
         * Return the index of the last frame accessed by the bgs.
         */
        int get_current_frame_index();

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgs;
        cv::VideoCapture m_input_video;
        std::string m_input_video_path;
        cv::Mat m_static_mask;
        int m_camera_index;
        int m_frame_index;
        int m_total_frame_count;
        float m_learning_rate;

        int m_starting_frame;
        int m_ending_frame;
        int m_step_size;

        void step();
        int count_neighbours_in_fg(cv::Mat frame, int x, int y, int buffer_size);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
