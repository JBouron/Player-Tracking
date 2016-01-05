#ifndef BACHELOR_PROJECT_CALIBRATION_TOOL_H
#define BACHELOR_PROJECT_CALIBRATION_TOOL_H

#include <string>
#include <opencv2/core/core.hpp>
#include "../background_subtractor/bgsubstractor.h"


namespace tmd{
    /* Class allowing the user to calibrate the BGS for each camera. */
    typedef enum {THRESHOLD_IDX=0, HISTORY_SIZE_IDX=1, LEARNING_RATE_IDX=2};
    class CalibrationTool{
    public:

        CalibrationTool(std::string video_folder_path,
                        std::string mask_folder_path);

        ~CalibrationTool();

        void calibrate();
        float**retrieve_params();

    private:
        float max(double a, double b);
        float min(double a, double b);
        std::string m_video_folder;
        cv::VideoCapture* m_videos[8];
        BGSubstractor* m_bgs[8];
        int m_current_camera;
        float m_params[8][3];
    };
}

#endif //BACHELOR_PROJECT_CALIBRATION_TOOL_H
