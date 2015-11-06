#ifndef BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H
#define BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H

#include <string>
#include "../headers/player_t.h"
#include "../headers/features_extractor.h"
#include "dpm_detector_tests.h"

namespace tmd {
    void features_extractor_tests(std::string image_path) {
        tmd::player_t *player = new tmd::player_t;
        player->original_image = cv::imread(image_path);
        const int rows = player->original_image.rows;
        const int cols = player->original_image.cols;
        player->mask_image = cv::Mat(rows, cols, CV_8U);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                player->mask_image.at<uchar>(i, j) = 255;
            }
        }
        std::string model = "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor"
                "-Project/misc/xmls/person.xml";
        tmd::FeaturesExtractor fe(model);
        fe.extractFeatures(player);

        // Show mask.
        cv::imshow("Mask", player->mask_image);
        cv::waitKey(0);

        // Test body parts.
        tmd::show_body_parts(player->original_image, player->features
                .body_parts);

        // Test histogram.
        cv::imshow("Histogram", player->features.color_histogram);
        cv::waitKey(0);

        // Check histogram values.
        cv::Mat hist = player->features.color_histogram;
        for (int i = 0; i < 180; i++) {
            float count = hist.at<float>(i);
            if (count != 0.f)
                std::cout << "Bin #" << i << " => " << count << std::endl;
        }


        //fe.~FeaturesExtractor();
        delete player;
    }
}

#endif //BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H