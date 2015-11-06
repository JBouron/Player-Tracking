#include <opencv2/imgproc/types_c.h>
#include "../headers/features_extractor.h"
#include "../headers/player_t.h"

namespace tmd{
    FeaturesExtractor::FeaturesExtractor(std::string model_file){
        m_detector = new DPMDetector(model_file);
        if (m_detector == NULL) {
            throw std::bad_alloc();
        }
    }

    FeaturesExtractor::~FeaturesExtractor() {
        delete m_detector;
    }

    void FeaturesExtractor::extractFeaturesFromPlayers(
            std::vector<player_t *> players) {
        size_t size = players.size();
        size_t i = 0;
        for (i = 0 ; i < size ; i ++){
            extractFeatures(players[i]);
        }
    }

    void FeaturesExtractor::extractFeatures(player_t *player) {
        if (player == NULL){
            throw std::invalid_argument("Error : Null pointer in "
                                                "FeaturesExtractor::extractFeatures()");
        }
        extractBodyParts(player);
        convertToHSV(player);
        updateMaskWithThreshold(player);
        createHistogram(player);
    }

    void FeaturesExtractor::extractBodyParts(player_t *p) {
        m_detector->extractBodyParts(p);
    }

    void FeaturesExtractor::convertToHSV(player_t *p) {
        cv::Mat hsv_tmp;
        cvtColor(p->original_image, hsv_tmp, CV_RGB2HSV);
        p->original_image = hsv_tmp;
    }

    bool isValid(double h, double s, double v){
        float th_red_low = TMD_FEATURE_EXTRACTOR_TH_RED_LOW;
        float th_red_high = TMD_FEATURE_EXTRACTOR_TH_RED_HIGH;
        float th_green_low = TMD_FEATURE_EXTRACTOR_TH_GREEN_LOW;
        float th_green_high = TMD_FEATURE_EXTRACTOR_TH_GREEN_HIGH;
        float th_sat_low = TMD_FEATURE_EXTRACTOR_TH_SATURATION_LOW;
        float th_val_low = TMD_FEATURE_EXTRACTOR_TH_VALUE_LOW;
        return ( (th_red_low < h && h <= 360) || (0 <= h && h <= th_red_high)||
                (th_green_low <= h && h <= th_green_high) ) &&
                (th_sat_low <= s) && (th_val_low <= v);

    }

    void FeaturesExtractor::updateMaskWithThreshold(player_t *p) {
        cv::Mat img = p->original_image;
        int cols = img.cols;
        int rows = img.rows;

        for (int i = 0 ; i < rows ; i ++){
            for (int j = 0 ; j < cols ; j ++){
                cv::Vec3b color = img.at<cv::Vec3b>(i, j);
                if (!isValid(color[0], color[1], color[2])){
                    // Need to 'remove' this pixel from the mask.
                    // TODO : Check the following line ...
                    p->mask_image.at<uchar>(i,j) = 0;
                }
            }
        }
    }

    void FeaturesExtractor::createHistogram(player_t *p) {

    }
}