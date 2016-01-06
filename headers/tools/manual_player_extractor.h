#ifndef BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H

#include "../players_extraction/player_extractor.h"
#include "../misc/debug.h"
#include "../data_structures/player_t.h"
#include "../data_structures/frame_t.h"

namespace tmd{
    /**
     * Manual PlayerExtractor, Used to test some functionalities.
     */
    class ManualPlayerExtractor : public PlayerExtractor{
    public:
        virtual std::vector<player_t*> extract_player_from_frame(frame_t* fr);

        static bool mBoxComplete;
        static bool mFirstClick;
        static std::vector<cv::Rect> mBoxes;
    private:
        static void onMouseClick(int event, int x, int y, int f, void* d);
    };
}

#endif //BACHELOR_PROJECT_MANUAL_PLAYER_EXTRACTOR_H
