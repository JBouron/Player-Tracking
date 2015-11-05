#ifndef TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H

#include <vector>
#include "player_t.h"
#include "dpm_detector.h"

namespace tmd {
    /* This class is responsible to extract the features from the players,
     * and update their feature member in the player_t struct.
     */

    class FeaturesExtractor {
    public:
        /**
         * Default constructor of the FeatureExtractor.
         * inputs :
         *      - model_file : path to the model file, given to the
         *      DPMDetector when crated.
         */
        FeaturesExtractor(std::string model_file);

        /**
         * Extract the features from a list of players by updating their
         * feature field.
         * inputs :
         *      - players : the list of players.
         */
        void extractFeaturesFromPlayers(std::vector<player_t *> players);

        /**
         * Extract features for one player.
         * Inputs :
         *      - player : The player to extract features from.
         */
        void extractFeatures(player_t *player);

    private:
        tmd::DPMDetector m_detector;

        /**
         * Helper method to extract the body part of a player.
         * Inputs :
         *      - p : The player to extract body parts from.
         */
        void extractBodyParts(player_t *p);

        /**
         * Convert the original_image field of a player from the RGB color
         * space to the HSV color space.
         * Input :
         *      - p : the player.
         */
        void convertToHSV(player_t *p);

        /**
         * Apply a hue threshold on the player image. The result is an
         * updated mask in the player given in parameter.
         */
        void updateMaskWithThreshold(player_t *p, float threshold);

        /**
         * Create the color histogram of the player given in parameter. The
         * histogram is then stored in the player features field.
         */
        void createHistogram(player_t *p);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
