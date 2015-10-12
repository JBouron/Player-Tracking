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
        static void extract_features_from_players(std::vector<player_t*> players);
    private:
        static void extract_features(player_t* player);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
