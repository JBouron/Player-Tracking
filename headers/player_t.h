#ifndef TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H
#define TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H

#include <opencv2/core/mat.hpp>
#include "position_t.h"
#include "team_t.h"
#include "box_t.h"

namespace tmd{
    /* Structure representing a player. */

    typedef struct{
        cv::Mat original_image;     // The original image of the player (when cropped from the frame).
        cv::Mat mask_image;         // The result of applying BGS on the original image. I gives us the important pixels.
        int frame_index;            // Frame indice from which this player has been extracted.
        tmd::position_t pos_frame;  // Position of the player on the frame.
        tmd::position_t pos_field;  // Position of the player on the field.
        tmd::team_t team;           // Team of the player (TBD by the TeamDecider).
        tmd::box_t box;             // Box of the player.
        float likelihood;           // Likelihood of the team membership.
        /** TODO : features of the player **/
    }player_t;
}

#endif //TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H

