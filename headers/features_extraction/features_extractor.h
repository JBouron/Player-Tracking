#ifndef TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H

#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../data_structures/player_t.h"
#include "../misc/debug.h"
#include "../misc/config.h"

/** Defines of the different threshold values used to create colors
 * histograms for the players.
 */

namespace tmd {

    /* This class is responsible to extract the features from the players,
     * and update their feature member in the player_t struct.
     */
    class FeaturesExtractor {
        /**
         * Create a friendship between this class and its corresponding test
         * class to make testing easier and keep helper methods private.
         */
        friend class FeaturesExtractorTest;

    public:
        /**
         * Extract the features from a list of players by updating their
         * feature field.
         * inputs :
         *      - players : the list of players.
         */
        void extractFeaturesFromPlayers(std::vector<player_t *> &players);

        /**
         * Extract features for one player.
         * Inputs :
         *      - player : The player to extract features from.
         */
        void extractFeatures(player_t *player);
        void createHistogram(player_t *p);

    private:
        /**
         * Convert the torso image field of a player from the RGB color
         * space to the HSV color space.
         * Input :
         *      - p : the player.
         */
        void convertToHSV(player_t *p);

        /**
         * Apply a hue threshold on the player image. The result is an
         * updated mask in the player given in parameter.
         */
        void updateMaskWithThreshold(player_t *p);

        /**
         * Create the color histogram of the player given in parameter. The
         * histogram is then stored in the player features field.
         */

        /**
         * Returns true if the given color is within the thresholds defined
         * above. False otherwise.
         */
        static bool withinThresholds(double h, double s, double v);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
