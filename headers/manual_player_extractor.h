#ifndef BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H

#include "player_extractor.h"
#include "../headers/debug.h"
#include "../headers/player_t.h"
#include "../headers/frame_t.h"

namespace tmd{
    class ManualPlayerExtractor : public PlayerExtractor{
    public:
        virtual std::vector<player_t*> extract_player_from_frame(frame_t* frame);

        static bool mBoxComplete;
        static bool mFirstClick;
        static std::vector<cv::Rect> mBoxes;
    private:
        static void onMouseClick(int event, int x, int y, int f, void* d);
    };
}

#endif //BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H
