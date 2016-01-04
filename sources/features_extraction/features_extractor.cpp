#include "../../headers/features_extraction/features_extractor.h"
#include "../../headers/data_structures/player_t.h"
#include "../../headers/data_structures/features_t.h"

namespace tmd {
    void FeaturesExtractor::extractFeaturesFromPlayers(
            std::vector<player_t *> &players) {
        size_t size = players.size();
        size_t i = 0;
        for (i = 0; i < players.size(); i++) {
            cv::imwrite("./res/debug/last_player_feature_extraction.jpg",
                        players[i]->original_image);
            extractFeatures(players[i]);
            if (players[i]->features.body_parts.size() == 0) {
                tmd::debug("FeaturesExtractor", "extractFeaturesFromPlayers",
                           "Player " + std::to_string(i) + " has no body "
                                   "parts !");
            }
        }
    }

    void FeaturesExtractor::extractFeatures(player_t *player) {
        if (player == NULL) {
            throw std::invalid_argument("Error : Null pointer in "
                                    "FeaturesExtractor::extractFeatures()");
        }
        //extractBodyParts(player);
        if (player->features.body_parts.size() > 0) {
            convertToHSV(player);
            //updateMaskWithThreshold(player);
            createHistogram(player);
        }
    }

    void FeaturesExtractor::convertToHSV(player_t *p) {
        cv::Mat hsv_tmp;
        cvtColor(p->features.torso, hsv_tmp, CV_BGR2HSV);
        p->features.torso = hsv_tmp;
    }

    bool FeaturesExtractor::withinThresholds(double h, double s, double v) {
        float th_red_low = Config::feature_extractor_threshold_red_low;
        float th_red_high = Config::feature_extractor_threshold_red_high;
        float th_green_low = Config::feature_extractor_threshold_green_low;
        float th_green_high = Config::feature_extractor_threshold_green_high;
        float th_sat_low = Config::feature_extractor_threshold_saturation;
        float th_val_low = Config::feature_extractor_threshold_value;

        return ((th_red_low < h && h <= 180) || (0 <= h && h <= th_red_high) ||
                (th_green_low <= h && h <= th_green_high)) &&
               (th_sat_low <= s) && (th_val_low <= v);

    }

    void FeaturesExtractor::updateMaskWithThreshold(player_t *p) {
        cv::Mat img = p->features.torso;
        int cols = img.cols;
        int rows = img.rows;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b color = img.at<cv::Vec3b>(i, j);
                bool in_mask = p->features.torso_mask.at<uchar>(i, j) > 127;
                if (withinThresholds(color[0], color[1], color[2]) && in_mask) {
                    p->features.torso_mask.at<uchar>(i, j) = 255;
                }
                else {
                    p->features.torso_mask.at<uchar>(i, j) = 0;
                }
            }
        }
    }

    void FeaturesExtractor::createHistogram(player_t *p) {
        if (p->features.torso_pos.width == 0 || p->features.torso_pos.height
                                                == 0){
            p->features.torso_color_histogram =
                cv::Mat(Config::feature_extractor_histogram_size, 1, CV_32F);
        }
        else{
            int bins_count = Config::feature_extractor_histogram_size;
            int dim = 1; // One dimension : The hue.
            float **range = new float *[1]; // freed
            range[0] = new float[2];
            range[0][0] = 0;
            range[0][1] = static_cast<float>(180);
            bool uniform = true;  // Make the histogram uniform.
            bool accumulate = false;
            std::vector<cv::Mat> imagechannels;
            cv::split(p->features.torso, imagechannels);
            cv::Mat images[] = {imagechannels[0]};
            std::vector<cv::Mat> maskchannels;
            cv::split(p->features.torso_mask, maskchannels);
            tmd::debug("FeaturesExtractor", "createHistogram",
                       "p->features.torso.channels() = " +
                       std::to_string(p->features.torso.channels()));
            tmd::debug("FeaturesExtractor", "createHistogram",
                       "p->features.torso_mask.channels() = " +
                       std::to_string(p->features.torso_mask.channels()));
            cv::calcHist(&images[0], 1, 0, maskchannels[0],
                         p->features.torso_color_histogram, dim, &bins_count,
                         (const float **) range,
                         uniform,
                         accumulate);
            cv::Mat histCpy = p->features.torso_color_histogram.clone();
            normalize(histCpy, p->features
                    .torso_color_histogram, 0, 1.0, cv::NORM_MINMAX, -1);
            delete[](range[0]);
            delete[](range);
        }
    }
}