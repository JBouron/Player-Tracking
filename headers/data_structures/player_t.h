#ifndef TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H
#define TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include "team_t.h"
#include "features_t.h"

namespace tmd{
    /* Structure representing a player. */

    typedef struct{
        cv::Mat original_image;     // The original image of the player
        cv::Mat mask_image;         // Resulting mask from the BGS. White
        // pixels are forground whereas black are from the background.
        int frame_index;            // Frame index containing this player.
        cv::Rect pos_frame;         // Position of the player on the frame.
        tmd::team_t team;           // Team of the player.
        float likelihood;           // Likelihood of the player detection.
                                    // (= DPM Score).
        features_t features;        // Features of the player.
    }player_t;

    /**
     * Helper function to release the memory taken by a player_t*.
     */
    inline void free_player(player_t* player){
        if (player != NULL){
            delete player;
        }
    }
}

#endif //TEAM_MEMBERSHIP_DETECTOR_PLAYER_T_H

