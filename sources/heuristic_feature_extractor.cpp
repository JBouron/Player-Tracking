
#include "../headers/heuristic_feature_extractor.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "../headers/player_t.h"
#include "../headers/debug.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

namespace tmd{
    void HeuristicFeaturesExtractor::extract_features_from_players(std::vector<player_t*> players){
        size_t i  = 0;
        size_t len = players.size();
        for (i = 0; i < len; i ++){
            extract_features(players[i]);
        }
    }

    void HeuristicFeaturesExtractor::extract_features(player_t *player) {
        cv::Mat playerImageResized;
        cv::Mat playerOriginalImage = player->original_image;
        double resizeFactorX = static_cast<double> (TMD_RESIZE_WIDTH) / playerOriginalImage.cols;
        double resizeFactorY = static_cast<double> (TMD_RESIZE_HEIGHT) / playerOriginalImage.rows;
        tmd::debug("HeuristicFeaturesExtractor", "extract_features", "resizeFactorX = " + std::to_string(resizeFactorX));
        tmd::debug("HeuristicFeaturesExtractor", "extract_features", "resizeFactorY = " + std::to_string(resizeFactorY));
        cv::resize(playerOriginalImage, playerImageResized, cv::Size(), resizeFactorX, resizeFactorY);

        tmd::debug("HeuristicFeaturesExtractor", "extract_features", "imax = " + std::to_string(TMD_RESIZE_HEIGHT / TMD_STRIP_WIDTH));
        for (int i = 0; i < TMD_RESIZE_HEIGHT / TMD_STRIP_WIDTH; i ++) {
            cv::Rect cropArea(0, i * TMD_STRIP_WIDTH, TMD_RESIZE_WIDTH, TMD_STRIP_WIDTH);
            player->features.strips.push_back(playerImageResized(cropArea));
        }
    }
}