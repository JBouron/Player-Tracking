#ifndef BACHELOR_PROJECT_HEURISTIC_FEATURE_EXTRACTOR_H
#define BACHELOR_PROJECT_HEURISTIC_FEATURE_EXTRACTOR_H

#include "player_t.h"
#include "features_extractor.h"
#include <vector>

#define TMD_RESIZE_WIDTH 200
#define TMD_RESIZE_HEIGHT 200
#define TMD_STRIP_WIDTH 100

namespace tmd{
    class HeuristicFeaturesExtractor : public FeaturesExtractor {
    public:
        virtual void extract_features_from_players(std::vector<player_t*> players);
        virtual void extract_features(player_t* player);
    };
}

#endif //BACHELOR_PROJECT_HEURISTIC_FEATURE_EXTRACTOR_H
