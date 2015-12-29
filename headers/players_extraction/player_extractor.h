#ifndef TEAM_MEMBERSHIP_DETECTOR_PLAYER_EXTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_PLAYER_EXTRACTOR_H

#include "../data_structures/frame_t.h"
#include "../data_structures/player_t.h"

namespace tmd{
    /* This class is responsible to extract the players from a given
     * frame. It will use the player extraction algorithm developed
     * by PlayfulVision.
     *
     * For each player (player_t), the extractor will update the following struct member :
     *      original_image
     *      mask_image
     *      frame_index
     *      pos_frame
     *
     * All the players for one frame are returned in a vector.
     */

    class PlayerExtractor{
    public:
        virtual std::vector<player_t*> extract_player_from_frame(frame_t* frame) = 0;
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_PLAYER_EXTRACTOR_H
