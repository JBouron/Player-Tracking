#ifndef TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
#define TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "player_t.h"
#include "dpm_detector.h"

/** Defines of the different threshold values used to create colors
 * histograms for the players.
 */
#define TMD_FEATURE_EXTRACTOR_TH_RED_LOW        (300 / 2.f)
#define TMD_FEATURE_EXTRACTOR_TH_RED_HIGH       (60 / 2.f)
#define TMD_FEATURE_EXTRACTOR_TH_GREEN_LOW      (100 / 2.f)
#define TMD_FEATURE_EXTRACTOR_TH_GREEN_HIGH     (140 / 2.f)
#define TMD_FEATURE_EXTRACTOR_TH_SATURATION_LOW 0.5f
#define TMD_FEATURE_EXTRACTOR_TH_VALUE_LOW      0.5f
#define TMD_FEATURE_EXTRACTOR_HISTOGRAM_SIZE    180

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
         * Default constructor of the FeatureExtractor.
         * inputs :
         *      - model_file : path to the model file, given to the
         *      DPMDetector when crated.
         */
        FeaturesExtractor(std::string model_file);

        /**
         * Destructor of the feature extractor.
         */
        ~FeaturesExtractor();

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
        tmd::DPMDetector *m_detector;

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
        void updateMaskWithThreshold(player_t *p);

        /**
         * Create the color histogram of the player given in parameter. The
         * histogram is then stored in the player features field.
         */
        void createHistogram(player_t *p);

        /**
         * Returns true if the given color is within the thresholds defined
         * above. False otherwise.
         */
        static bool withinThresholds(double h, double s, double v);
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_FEATURES_EXTRACTOR_H
