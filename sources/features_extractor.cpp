#include <opencv2/imgproc/types_c.h>
#include "../headers/features_extractor.h"
#include "../headers/debug.h"

namespace tmd {

    FeaturesExtractor::FeaturesExtractor(std::string model_file) {
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
        for (i = 0; i < size; i++) {
            extractFeatures(players[i]);
        }
    }

    void FeaturesExtractor::extractFeatures(player_t *player) {
        if (player == NULL) {
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
        cvtColor(p->original_image, hsv_tmp, CV_BGR2HSV);
        p->original_image = hsv_tmp;
    }

    bool FeaturesExtractor::withinThresholds(double h, double s, double v){
        float th_red_low = TMD_FEATURE_EXTRACTOR_TH_RED_LOW;
        float th_red_high = TMD_FEATURE_EXTRACTOR_TH_RED_HIGH;
        float th_green_low = TMD_FEATURE_EXTRACTOR_TH_GREEN_LOW;
        float th_green_high = TMD_FEATURE_EXTRACTOR_TH_GREEN_HIGH;
        float th_sat_low = TMD_FEATURE_EXTRACTOR_TH_SATURATION_LOW;
        float th_val_low = TMD_FEATURE_EXTRACTOR_TH_VALUE_LOW;
        tmd::debug("FeaturesExtractor", "isValid", "h = " + std::to_string(h)
                                                   + "s = " + std::to_string(s)
                                                   + "v = " +
                                                   std::to_string(v));
        return ((th_red_low < h && h <= 360) || (0 <= h && h <= th_red_high) ||
                (th_green_low <= h && h <= th_green_high)) &&
               (th_sat_low <= s) && (th_val_low <= v);

    }

    void FeaturesExtractor::updateMaskWithThreshold(player_t *p) {
        cv::Mat img = p->original_image;
        int cols = img.cols;
        int rows = img.rows;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b color = img.at<cv::Vec3b>(i, j);
                if (withinThresholds(color[0], color[1], color[2])){
                    p->mask_image.at<uchar>(i,j) = 255;
                }
                else{
                    p->mask_image.at<uchar>(i,j) = 0;
                }
            }
        }
    }

    void FeaturesExtractor::createHistogram(player_t *p) {
        int bins_count = 180;
        int dim = 1; // One dimension : The hue.
        float **range = new float *[1];
        range[0] = new float[2];
        range[0][0] = 0;
        range[0][1] = static_cast<float>(180);
        bool uniform = true;  // Make the histogram uniform.
        bool accumulate = false;
        std::vector<cv::Mat> imagechannels;
        cv::split(p->original_image, imagechannels);
        cv::Mat images[] = {imagechannels[0]};
        std::vector<cv::Mat> maskchannels;
        cv::split(p->mask_image, maskchannels);
        tmd::debug("FeaturesExtractor", "createHistogram", "p->original_image"
                                                                   ".channels() = " +
                                                           std::to_string(
                                                                   p->original_image.channels
                                                                           ()));
        tmd::debug("FeaturesExtractor", "createHistogram", "p->mask_image"""
                                                                   ".channels() = " +
                                                           std::to_string(
                                                                   p->original_image.channels()));
        cv::calcHist(&images[0], 1, 0, maskchannels[0], p->features
                             .torso_color_histogram, dim, &bins_count,
                     (const float **) range,
                     uniform,
                     accumulate);
    }
}