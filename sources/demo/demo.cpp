#include <iostream>
#include <stack>
#include "../../headers/demo/demo.h"
#include "../../headers/features_extractor.h"
#include "../../headers/bgsubstractor.h"
#include "../../headers/feature_comparator.h"
#include "../../headers/player_extractor.h"
#include "../../headers/manual_player_extractor.h"
#include "../../headers/player_t.h"
#include "../../headers/features_t.h"

namespace tmd {

    void run_demo_feature_comparator(void) {
        std::string video_folder_green = "./res/videos/alone-green-no-ball";
        std::string video_folder_red = "./res/videos/alone-red-no-ball";
        std::string mask_folder = "./res/bgs_masks";

        cv::VideoCapture videos_green[8];
        cv::VideoCapture videos_red[8];
        std::string masks[8];
        for (int i = 0; i < 8; i++) {
            std::string path = video_folder_green + "/ace_" + std::to_string(i) + ".mp4";
            videos_green[i].open(path);
            path = video_folder_red + "/ace_" + std::to_string(i) + ".mp4";
            videos_red[i].open(path);
            std::string mask_path = mask_folder + "/mask_ace" + std::to_string(i) + ".jpg";
            masks[i] = mask_path;

        }

        tmd::BGSubstractor *bgs_green[8];
        tmd::BGSubstractor *bgs_red[8];
        for (unsigned char i = 0; i < 8; i++) {
            bgs_green[i] = new tmd::BGSubstractor(&videos_green[i],
                                                  cv::imread(masks[i], 0), i,1);
            bgs_red[i] = new tmd::BGSubstractor(&videos_red[i], cv::imread
                    (masks[i], 0), i, 1);
        }

        int clusterCols = 180;
        int clusterCount = 2;
        cv::Mat clusterCenters = cv::Mat(clusterCount, clusterCols, CV_32F);
        tmd::FeatureComparator comparator(clusterCount, clusterCols, clusterCenters);

        tmd::ManualPlayerExtractor playerExtractor;
        tmd::FeaturesExtractor featuresExtractor("./res/xmls/person.xml");

        std::stack<player_t *> players;


        for (int i = 0; i < 8; i++) {
            bgs_red[i]->jump_to_frame(600);
            bgs_green[i]->jump_to_frame(600);
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 10; k++) {
                    bgs_green[i]->next_frame();
                    bgs_red[i]->next_frame();
                }
                std::vector<player_t *> playersExtractedGreen = playerExtractor.extract_player_from_frame(
                        bgs_green[i]->next_frame());
                std::vector<player_t *> playersExtractedRed = playerExtractor.extract_player_from_frame(
                        bgs_red[i]->next_frame());

                for (int l = 0; l < playersExtractedGreen.size(); l++) {
                    players.push(playersExtractedGreen[l]);
                    featuresExtractor.extractFeatures(players.top());
                    comparator.addPlayerFeatures(players.top());
                    delete players.top();
                }
                for (int l = 0; l < playersExtractedRed.size(); l++) {
                    players.push(playersExtractedRed[l]);
                    featuresExtractor.extractFeatures(players.top());
                    comparator.addPlayerFeatures(players.top());
                    delete players.top();
                }

            }
        }

        comparator.runClustering();
        comparator.writeCentersToFile();


        std::string video_folder_two_green = "./res/videos/two-green-no-ball";
        std::string video_folder_two_red = "./res/videos/two-red-no-ball";

        cv::VideoCapture videos_two_green[8];
        cv::VideoCapture videos_two_red[8];
        for (int i = 0; i < 8; i++) {
            std::string path = video_folder_green + "/ace_" + std::to_string(i) + ".mp4";
            videos_two_green[i].open(path);
            path = video_folder_red + "/ace_" + std::to_string(i) + ".mp4";
            videos_two_red[i].open(path);
        }

        cv::Mat readCenters = tmd::FeatureComparator::readCentersFromFile(clusterCount,
                                                      clusterCols);
        cv::Mat center1 = readCenters.row(0);
        cv::Mat center2 = readCenters.row(1);

        std::cout << "read from file :" << std::endl;
        for (int i = 0; i < 180; i++) {
            std::cout << readCenters.at<float>(0, i) << ", ";
        }
        std::cout << std::endl;

        tmd::BGSubstractor *bgs_two_green[8];
        tmd::BGSubstractor *bgs_two_red[8];
        for (unsigned char i = 0; i < 8; i++) {
            bgs_two_green[i] = new tmd::BGSubstractor(&videos_two_green[i],
                                                      cv::imread(masks[i], 0)
                    , i, 1);
            bgs_two_red[i] = new tmd::BGSubstractor(&videos_two_red[i],
                                                    cv::imread(masks[i], 0),
                                                    i, 1);
        }

        for (int i = 0; i < 8; i++) {
            bgs_two_red[i]->jump_to_frame(600);
            bgs_two_green[i]->jump_to_frame(600);
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 10; k++) {
                    bgs_two_green[i]->next_frame();
                    bgs_two_red[i]->next_frame();
                }

                std::vector<player_t *> playersExtractedGreen = playerExtractor.extract_player_from_frame(
                        bgs_two_green[i]->next_frame());
                for (int l = 0; l < playersExtractedGreen.size(); l++) {
                    featuresExtractor.extractFeatures(playersExtractedGreen[l]);
                    /*cv::Mat closest = comparator.getClosestCenter
                            (playersExtractedGreen[i]);
                    compareCenters(center1, closest);*/
                }

                std::vector<player_t *> playersExtractedRed = playerExtractor.extract_player_from_frame(
                        bgs_two_red[i]->next_frame());
                for (int l = 0; l < playersExtractedRed.size(); l++) {
                    featuresExtractor.extractFeatures(playersExtractedRed[l]);
                    /*cv::Mat closest = comparator.getClosestCenter
                            (playersExtractedRed[i]);
                    compareCenters(center1, closest);*/
                }
            }
        }

    }

    void compareCenters(cv::Mat center1, cv::Mat compare) {
        bool equal = true;
        for (int i = 0; i < 180; i++) {
            if (center1.at<float>(0, i) != compare.at<float>(0, i)) {
                equal = false;

            }
        }
        if (equal) {
            std::cout << "CENTER 1" << std::endl;
        }
        else {
            std::cout << "CENTER 2" << std::endl;
        }
    }

    void run_demo_dpm(void) {
        const int player_count = 5;
        player_t **players = new player_t *[player_count];
        for (int i = 0; i < player_count; i++) {
            players[i] = new player_t;
            std::string path = "./res/demo/dpm/img" + std::to_string(i) + ".jpg";
            std::cout << path << std::endl;
            players[i]->original_image = cv::imread(path);
            player_t *player = players[i];
            player->pos_frame.x = 0;
            player->pos_frame.y = 0;
            player->pos_frame.width = player->original_image.cols;
            player->pos_frame.height = player->original_image.rows;
            const int rows = player->original_image.rows;
            const int cols = player->original_image.cols;
            player->mask_image = cv::Mat(rows, cols, CV_8U);
            for (int l = 0; l < rows; l++) {
                for (int j = 0; j < cols; j++) {
                    player->mask_image.at<uchar>(l, j) = 255;
                }
            }
        }
        DPMDetector dpm("./res/xmls/person.xml");
        for (int i = 0; i < player_count; i++) {
            std::cout << "image " << i << " ";
            dpm.extractBodyParts(players[i]);
            std::cout << "done" << std::endl;
            cv::Rect torsoPos = players[i]->features.torso_pos;
            cv::Mat torso = players[i]->original_image(torsoPos);
            cv::imwrite("./res/demo/dpm/torso" + std::to_string(i) + ".jpg",
                        torso);
        }

        for (int i = 0; i < player_count; i++) {
            show_dpm_detection_parts(players[i]);
        }

        for (int i = 0; i < player_count; i++) {
            delete players[i];
        }
        delete players;
    }

    void run_demo_pipeline(void) {
        tmd::player_t *player = new player_t;
        player->original_image = cv::imread("./res/manual_extraction/playeror"
                                                    ".jpg");
        cv::Mat mask = cv::imread("./res/manual_extraction/playermk"
                                          ".jpg");
        const int rows = player->original_image.rows;
        const int cols = player->original_image.cols;
        player->mask_image = cv::Mat(rows, cols, CV_8U);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b s = mask.at<cv::Vec3b>(i, j);
                std::cout << s.val[0] << " " << s.val[1] << " " << s.val[2]
                << std::endl;
                if (s.val[1] > 127) {
                    player->mask_image.at<uchar>(i, j) = 255;
                }
                else {
                    player->mask_image.at<uchar>(i, j) = 0;
                }
            }
        }

        // Extract features from the player.
        tmd::FeaturesExtractor featuresExtractor("./res/xmls/person.xml");
        featuresExtractor.extractFeatures(player);

        // Show player image and mask.
        show_original_image(player);
        show_original_image_and_mask(player);

        // Show every parts of the filter.
        show_dpm_detection_parts(player);

        // Show torso box of the player.
        show_torso_part(player);

        // Show the mask for the torso before Hue threshold.
        show_torso_mask_before_th(player);

        // Show the mask for the torso after Hue threshold.
        show_torso_mask_after_th(player);

        // Create Color Histogram for the torso.
        show_torso_histogram(player);

    }

    void show_original_image(const tmd::player_t *const player) {
        cv::imshow("Original player image", player->original_image);
        cv::waitKey(0);
        cv::destroyWindow("Original player image");
    }

    void show_original_image_and_mask(const tmd::player_t *const player) {
        cv::imshow("Original mask image image form BGS", player->mask_image);
        cv::waitKey(0);
        cv::destroyWindow("Original mask image image form BGS");
    }

    void show_dpm_detection_parts(const tmd::player_t *const player) {
        std::string win_name = "DPM Detection parts";
        cv::Mat partsimg = player->original_image.clone();
        std::vector<cv::Rect> parts = player->features.body_parts;
        CvScalar color;
        color.val[0] = 255;
        color.val[1] = 0;
        color.val[2] = 255;
        color.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        for (int i = 0; i < parts.size(); i++) {
            CvRect r;
            r.x = parts[i].x;
            r.y = parts[i].y;
            r.width = parts[i].width;
            r.height = parts[i].height;
            cv::rectangle(partsimg, r, color, thickness, line_type, shift);
        }
        cv::imshow(win_name, partsimg);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }

    void show_torso_part(const tmd::player_t *const player) {
        std::string win_name = "Find the torso";
        cv::Mat torsoimg = player->original_image.clone();
        CvScalar torso;
        torso.val[0] = 255;
        torso.val[1] = 255;
        torso.val[2] = 0;
        torso.val[3] = 255;
        CvRect r;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        r.x = player->features.torso_pos.x;
        r.y = player->features.torso_pos.y;
        r.width = player->features.torso_pos.width;
        r.height = player->features.torso_pos.height;
        cv::rectangle(torsoimg, r, torso, thickness, line_type, shift);
        cv::imshow(win_name, torsoimg);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }

    void show_torso_mask_before_th(const tmd::player_t *const player) {
        cv::Mat torso_mask_before = (player->mask_image(player->features
                                                                .torso_pos));
        std::string win_name = "Torso mask before hue threshold";
        cv::imshow(win_name, torso_mask_before);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }

    void show_torso_mask_after_th(const tmd::player_t *const player) {
        std::string win_name = "Torso mask after hue threshold";
        cv::imshow(win_name, player->features.torso_mask);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }

    void show_torso_histogram(const tmd::player_t *const player) {
        cv::Mat localHist = player->features.torso_color_histogram.clone();
        // Normalize the histogram ...
        const int hist_h = TMD_FEATURE_EXTRACTOR_HISTOGRAM_SIZE;
        const int hist_w = TMD_FEATURE_EXTRACTOR_HISTOGRAM_SIZE;
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        normalize(localHist, localHist, 0, 1, cv::NORM_MINMAX, -1,
                  cv::Mat());
        std::cout << localHist << std::endl;
        for (int i = 1; i < TMD_FEATURE_EXTRACTOR_HISTOGRAM_SIZE; i++) {
            line(histImage, cv::Point(1 * (i - 1),
                                      hist_h - cvRound(localHist.at<float>(i - 1))),
                 cv::Point(1 * (i), hist_h - cvRound(localHist.at<float>
                         (i))),
                 cv::Scalar(255, 0, 0), 1, 8, 0);
        }
        std::string win_name = "Color Histogram for the torso after hue "
                "threshold";
        cv::imshow(win_name, histImage);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }
}