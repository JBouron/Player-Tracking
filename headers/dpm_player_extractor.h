#ifndef BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H

#include <opencv2/objdetect/objdetect.hpp>
#include "player_extractor.h"

namespace tmd{

    /**
     * Player extractor using DPM to detect and crop players from the frame.
     */
    class DPMPlayerExtractor : public tmd::PlayerExtractor{
    public :
        DPMPlayerExtractor(std::string model_file);
        virtual std::vector<player_t*> extract_player_from_frame(frame_t*
        frame);

    private:
        cv::LatentSvmDetector* m_detector;
    };
}

#endif //BACHELOR_PROJECT_DPM_PLAYER_EXTRACTOR_H
