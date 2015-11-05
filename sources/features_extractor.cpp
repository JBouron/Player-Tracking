#include "../headers/features_extractor.h"

namespace tmd{
    FeaturesExtractor::FeaturesExtractor(std::string model_file){
        m_detector = new DPMDetector(model_file);
        if (m_detector == NULL) {
            throw new std::bad_alloc;
        }
    }

    FeaturesExtractor::~FeaturesExtractor() {

    }

    void FeaturesExtractor::extractFeaturesFromPlayers(
            std::vector<player_t *> players) {

    }

    void FeaturesExtractor::extractFeatures(player_t *player) {

    }

    void FeaturesExtractor::extractBodyParts(player_t *p) {

    }

    void FeaturesExtractor::convertToHSV(player_t *p) {

    }

    void FeaturesExtractor::updateMaskWithThreshold(player_t *p,
                                                    float threshold) {

    }

    void FeaturesExtractor::createHistogram(player_t *p) {

    }
}