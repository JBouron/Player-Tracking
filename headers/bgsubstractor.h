#ifndef TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H

#include <exception>
#include <stdexcept>
#include <opencv2/highgui/highgui.hpp>
#include "frame_t.h"

#define TMD_BGS_DETECTS_SHADOWS false

namespace tmd {
    /* Class responsible of applying a BG substraction on a given video.
     * The BGS works as an iterator over the input video. You can extract
     * frames from the video one by one and apply a background substraction
     * on them.
     */

    class BGSubstractor {
    public:
        /**
         * Constructor of the Background Substractor.
         * input_video : The video to operate on.
         * static_mask : The static to use.
         * camera_index : The index of the camera.
         * step_size : The step size between to frames.
         * threshold : Threshold for the color distance to use during
         *              computation.
         * history : Size of the history to keep for the background.
         * learning_rate : Learning rate of the algorithm, ie. how does the
         * background model changes over time.
         */
        BGSubstractor(cv::VideoCapture *input_video, cv::Mat static_mask,
                      unsigned char camera_index, int step_size = 1,
                      float threshold = 256, int history = 500,
                      float learning_rate = 0.0);

        /**
         * Destructor of the BGS.
         */
        ~BGSubstractor();

        /**
         * Extract the next frame from the input_video.
         * Return NULL if there is no frame left in the input stream.
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
         * Set the step size of this BGS.
         */
        void set_step_size(int step);

        /**
         * Return the index of the last frame accessed by the bgs.
         */
        int get_current_frame_index();

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgs;
        cv::VideoCapture *m_input_video;
        cv::Mat m_static_mask;
        unsigned char m_camera_index;
        double m_frame_index;
        double m_total_frame_count;
        float m_learning_rate;
        int m_step_size;

        int count_neighbours_in_fg(cv::Mat frame, int x, int y, int buffer_size);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_BGSUBSTRACTOR_H
