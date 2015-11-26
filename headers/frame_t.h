#ifndef TEAM_MEMBERSHIP_DETECTOR_FRAME_T_H
#define TEAM_MEMBERSHIP_DETECTOR_FRAME_T_H

#include <opencv2/core/core.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace tmd {

    /* Structure frame_t.
     * The role of the frame_t structure is to hold relevant information of a frame
     * taken from the input video.
    */

    typedef struct {
        cv::Mat original_frame;         // Original frame taken from the video.
        double frame_index;             // Index of the frame in the video.
        cv::Mat mask_frame;             // Frame after applying background substraction.
        unsigned char camera_index;     // Index of the camera which took the frame.
    } frame_t;
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FRAME_T_H
