#ifndef TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H

#include <vector>
#include "player_t.h"

namespace tmd{
    /* This class is responsible to extract the features from the players,
     * and update their feature member in the player_t struct.
     */

    class FeaturesExtractor{
    public:
        virtual void extract_features_from_players(std::vector<player_t*> players) = 0;
        virtual void extract_features(player_t* player) = 0;
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
