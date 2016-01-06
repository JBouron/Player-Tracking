#include "../../headers/tools/dpm_calibrator.h"

namespace tmd {

    void apply_mask_on_frame(frame_t *frame) {
        cv::Vec3b black;
        black.val[0] = 0;
        black.val[1] = 0;
        black.val[2] = 0;
        cv::Vec3b white;
        white.val[0] = 255;
        white.val[1] = 255;
        white.val[2] = 255;
        for (int c = 0; c < frame->mask_frame.cols; c++) {
            for (int r = 0; r < frame->mask_frame.rows; r++) {
                if (frame->mask_frame.at<uchar>(r, c) == 0) {
                    frame->original_frame.at<cv::Vec3b>(r, c) = black;
                }
            }
        }
    }

    void DPMCalibrator::calibrate_dpm(std::string video_path,
                                      std::string mask_path, int start_frame,
                                      int frame_step) {
        cv::VideoCapture capture(video_path);
        if (!capture.isOpened()) {
            throw std::invalid_argument("Error in DPM Calibrator : No such "
                                                "video file found");
        }

        CvScalar color;
        color.val[0] = 255;
        color.val[1] = 0;
        color.val[2] = 255;
        color.val[3] = 255;
        CvScalar torso;
        torso.val[0] = 255;
        torso.val[1] = 255;
        torso.val[2] = 0;
        torso.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;

        DPMPlayerExtractor dpmPlayerExtractor;
        BGSubstractor bgSubstractor(video_path, 0);
        FeaturesExtractor featuresExtractor;

        int keyboard = 0;

        bgSubstractor.jump_to_frame(start_frame);

        frame_t *frame = bgSubstractor.next_frame();
        apply_mask_on_frame(frame);

        cv::Mat frame_cpy(frame->original_frame);

        bool recompute_needed = false;

        float score_threshold = dpmPlayerExtractor.get_score_threshold();
        float overlapping_threshold = dpmPlayerExtractor
                .get_overlapping_threshold();

        std::string win_name = "DPM Calibration";

        std::vector<tmd::player_t *> players;

        while (keyboard != 27 && frame != NULL) {
            // Next frame requested
            recompute_needed = true;
            switch (keyboard) {
                case 'n' :
                    delete frame;
                    for (int i = 0; i < frame_step; i++) {
                        delete bgSubstractor.next_frame();
                    }
                    frame = bgSubstractor.next_frame();
                    apply_mask_on_frame(frame);
                    break;

                case 'o': // Increase overlapping threshold.
                    overlapping_threshold += 0.1;
                    break;

                case 'l': // Decrease overlapping threshold.
                    overlapping_threshold -= 0.1;
                    break;

                case 's': // Increase score threshold.
                    score_threshold += 0.1;
                    recompute_needed = false;
                    std::cout << "score_threshold = " << score_threshold <<
                    std::endl;
                    break;

                case 'x': // Decrease score threshold.
                    score_threshold -= 0.1;
                    recompute_needed = false;
                    std::cout << "score_threshold = " << score_threshold <<
                    std::endl;
                    break;

                default:
                    recompute_needed = false;
                    break;
            }

            if (recompute_needed) {
                // Free the player vector.
                for (size_t i = 0; i < players.size(); i++) {
                    delete players[i];
                }
                players.clear();

                std::cout << "Frame " << bgSubstractor.
                        get_current_frame_index() << std::endl;

                dpmPlayerExtractor.set_overlapping_threshold
                        (overlapping_threshold);
                dpmPlayerExtractor.set_score_threshold(score_threshold);

                // Extract players from the frame.
                players = dpmPlayerExtractor
                        .extract_player_from_frame(frame);

                std::cout << "    " << players.size() << " players detected." <<
                std::endl;

                // For each player
                for (size_t i = 0; i < players.size(); i++) {
                    featuresExtractor.extractFeatures(players[i]);
                }

                std::cout << "overlapping_threshold = " <<
                overlapping_threshold << std::endl;

                std::cout << "score_threshold = " << score_threshold <<
                std::endl;

            }
            frame_cpy = frame->original_frame.clone();

            // For each player
            for (size_t i = 0; i < players.size(); i++) {
                tmd::player_t *player = players[i];
                if (player->likelihood > score_threshold) {
                    // Draw detection rectangle
                    cv::rectangle(frame_cpy, player->pos_frame, color,
                                  thickness,
                                  line_type, shift);

                    // Draw the parts rectangles
                    for (size_t j = 0;
                         j < player->features.body_parts.size(); j++) {

                        cv::Rect part = player->features.body_parts[j];
                        part.x += player->pos_frame.x;
                        part.y += player->pos_frame.y;
                        cv::rectangle(frame_cpy, part, torso, thickness,
                                      line_type,
                                      shift);
                    }
                }
            }

            cv::imshow(win_name, frame_cpy);
            keyboard = cv::waitKey(0);
        }
        cv::destroyWindow(win_name);
    }
}