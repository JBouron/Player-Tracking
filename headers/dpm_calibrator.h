#ifndef BACHELOR_PROJECT_DPM_CALIBRATOR_H
#define BACHELOR_PROJECT_DPM_CALIBRATOR_H

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../headers/dpm_player_extractor.h"
#include "../headers/bgsubstractor.h"
#include "../headers/features_extractor.h"

namespace tmd{

    /**
     * Class helping us to determine the effect of each thresholds/parameter
     * used by the dpm detector.
     */
    class DPMCalibrator{
    public:
        /**
         * Launch the calibration on the given video.
         */
        static void calibrate_dpm(std::string video_path, std::string mask_path, int start_frame,
                                  int frame_step);
    };
}

#endif //BACHELOR_PROJECT_DPM_CALIBRATOR_H
