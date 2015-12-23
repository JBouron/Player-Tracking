#include <iostream>
#include "../headers/frame_t.h"
#include "../headers/test_cases/test_suite.h"
#include "../headers/demo/demo.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/calibration_tool.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/features_extractor.h"
#include "../headers/feature_comparator.h"
#include "../headers/dpm_calibrator.h"
#include "../headers/pipeline.h"
#include "../headers/training_set_creator.h"
#include "../headers/blob_separator.h"
#include "../headers/pipeline_visualizer.h"

void show_body_parts(cv::Mat image, tmd::player_t *p);

void extract_player_image(void);

void dpm_feature_extractor_test(void);

void pipeline(void);

void pipeline_class_tests(void);

void dpm_whole_frame(void);

void test_blob_separation(void);

void memleak_video_capture(void) {
    cv::VideoCapture *capture = new cv::VideoCapture();
    capture->open("./res/videos/alone-green-ball/ace_0.mp4");
    for (int i = 0; i < 1000; i++) {
        cv::Mat frame;
        capture->read(frame);
    }
    capture->release();
    delete capture;
    std::cout << "Freed" << std::endl;
}

int main(int argc, char *argv[]) {
    tmd::Pipeline pipeline("./res/videos/uni-hockey/ace_0.mp4", ""
                                   "./res/bgs_masks/mask_ace0.jpg", 0, ""
                                   "./res/xmls/person.xml", false, true,
                           "./res/pipeline_results/complete_pipeline/uni/with blob separator/");

    pipeline.set_frame_step_size(10);
    pipeline.set_start_frame(0);

    tmd::frame_t *frame = pipeline.next_frame();
    tmd::PipelineVisualizer visualizer(&pipeline,
            "./res/videos/uni-hockey/ace_0.mp4", 10);
    visualizer.run();
    return 0;

    double t1 = cv::getTickCount();
    int count = 0;
    int max_frames = -1;
    while (frame != NULL) {
        tmd::free_frame(frame);
        frame = pipeline.next_frame();
        count++;
        if (count == max_frames) {
            break;
        }
    }
    double t2 = cv::getTickCount();
    std::cout << "Time = " << (t2 - t1) / cv::getTickFrequency() << std::endl;
    return EXIT_SUCCESS;
}

void test_blob_separation(void) {
    tmd::player_t *player = new tmd::player_t;
    player->original_image = cv::imread(
            "./res/manual_extraction/frame5847_originalimage0.jpg");
    player->mask_image = cv::imread(
            "./res/manual_extraction/frame5847_maskimage0.jpg", 0);
    player->frame_index = 0;

    std::vector<tmd::player_t *> players;
    players.push_back(player);

    std::vector<tmd::player_t *> new_players = tmd::BlobSeparator::separate_blobs
            (players);

    std::cout << "new_player size = " << new_players.size() << std::endl;

    for (int i = 0; i < new_players.size(); i++) {
        cv::imshow("Player", new_players[i]->original_image);
        cv::waitKey(0);
    }
}

void create_training_set(void) {

    std::string basic_path = "./res/videos/";

    std::string video_folders[8];
    std::string mask_folder[6];

    video_folders[0] = basic_path + "alone-green-ball/";
    video_folders[1] = basic_path + "alone-green-no-ball/";
    video_folders[2] = basic_path + "alone-red-ball/";
    video_folders[3] = basic_path + "alone-red-no-ball/";
    video_folders[4] = basic_path + "two-green-ball/";
    video_folders[5] = basic_path + "two-green-no-ball/";
    video_folders[6] = basic_path + "two-red-ball/";
    video_folders[7] = basic_path + "two-red-no-ball/";

    cv::VideoCapture videos[48];

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            std::string video_path = video_folders[i] + "ace_" + std::to_string(j) + ".mp4";
            videos[i + j].open(video_path);
        }
    }

    cv::Mat centers;
    tmd::FeatureComparator *comparator = new tmd::FeatureComparator(2, 180, centers);
}

void pipeline_class_tests(void) {
    tmd::Pipeline pipeline("./res/videos/alone-green-no-ball/ace_0.mp4", "./res/bgs_masks/mask_ace0.jpg", 0, ""
            "./res/xmls/person.xml", false, true, ""
                                   "./res/pipeline_results/complete_pipeline/alone-green-no-ball/");

    pipeline.set_frame_step_size(10);
    pipeline.set_start_frame(0);

    pipeline.set_end_frame(1200);

    int keyboard = 0;
    std::string win_name = "Pipeline frame";
    tmd::frame_t *frame = pipeline.next_frame();
    while (keyboard != 27 && frame != NULL) {
        cv::imshow(win_name, frame->original_frame);
        keyboard = cv::waitKey(0);

        if (keyboard == 'n') {
            delete frame;
            frame = pipeline.next_frame();
        }
    }
    cv::destroyWindow(win_name);
    tmd::free_frame(frame);
}

void dpm_whole_frame(void) {
    tmd::player_t *player = new tmd::player_t;
    player->original_image = cv::imread("./res/images/uni0.jpg");
    const int rows = player->original_image.rows;
    const int cols = player->original_image.cols;
    player->mask_image = cv::Mat(rows, cols, CV_8U);
    for (int l = 0; l < rows; l++) {
        for (int j = 0; j < cols; j++) {
            player->mask_image.at<uchar>(l, j) = 255;
        }
    }

    tmd::DPMDetector dpmDetector("./res/xmls/person.xml");
    dpmDetector.extractBodyParts(player);
    show_body_parts(player->original_image, player);
}

void extract_player_image(void) {
    cv::VideoCapture capt("./res/videos/alone-green-no-ball/ace_0.mp4");
    tmd::BGSubstractor bgs(&capt, cv::imread("./res/bgs_masks/mask_ace0.jpg", 0
    ), 0);
    bgs.jump_to_frame(800);
    int keyboard = 0;
    cv::namedWindow("Extraction");
    tmd::frame_t *frame;
    while (keyboard != 27) {
        keyboard = cv::waitKey(15);
        frame = bgs.next_frame();
        cv::imshow("Extraction", frame->original_frame);
        if (keyboard != 27) {
            delete frame;
        }
    }
    tmd::ManualPlayerExtractor mp;
    mp.extract_player_from_frame(frame);
}

void show_body_parts(cv::Mat image, tmd::player_t *p) {
    std::vector<cv::Rect> parts = p->features.body_parts;
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
    cv::Mat destImg = image.clone();
    std::cout << "Parts count = " << parts.size() << std::endl;
    for (int i = parts.size(); i > 0; i--) {
        if (i % 6 == 0) {
            std::cout << "i = " << i << std::endl;
            destImg = image.clone();
        }
        CvRect r;
        r.x = parts[i].x;
        r.y = parts[i].y;
        r.width = parts[i].width;
        r.height = parts[i].height;
        cv::rectangle(destImg, r, color, thickness, line_type, shift);
        cv::imshow("Body parts", destImg);
        cv::waitKey(0);
    }
}

void pipeline(void) {
    cv::VideoCapture capture("./res/videos/alone-green-no-ball/ace_0.mp4");
    tmd::BGSubstractor bgSubstractor(&capture, cv::imread("./res/bgs_masks/mask_ace0.jpg"), 0);
    tmd::DPMPlayerExtractor dpmPlayerExtractor("./res/xmls/person.xml");
    tmd::FeaturesExtractor featuresExtractor("./res/xmls/person.xml");

    const int frame_start = 610;
    const int frame_limit = 800;
    const int frame_step = 10;
    std::vector<cv::Mat> frames_results;
    int frame_idx = frame_start;

    bgSubstractor.jump_to_frame(frame_start);

    const bool show_intermediate_results = true;
    const bool save_results = false;

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

    while (frame_idx < frame_limit) {
        std::cout << "In frame " << frame_idx << " : " << std::endl;

        // Fetch next frame.
        tmd::frame_t *frame = bgSubstractor.next_frame();

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
        // Extract players from the frame.
        std::vector<tmd::player_t *> players = dpmPlayerExtractor
                .extract_player_from_frame(frame);
        std::cout << players.size() << " players detected." << std::endl;

        cv::Mat frame_cpy(frame->original_frame);
        // For each player
        for (size_t i = 0; i < players.size(); i++) {
            tmd::player_t *player = players[i];
            // Draw detection rectangle
            cv::rectangle(frame_cpy, player->pos_frame, color, thickness,
                          line_type, shift);

            // Extract the features.
            featuresExtractor.extractFeatures(player);

            // Draw the parts rectangles
            for (size_t j = 0; j < player->features.body_parts.size(); j++) {
                cv::Rect part = player->features.body_parts[j];
                part.x += player->pos_frame.x;
                part.y += player->pos_frame.y;
                cv::rectangle(frame_cpy, part, torso, thickness, line_type,
                              shift);
            }
        }

        // Display the result in the window.
        if (show_intermediate_results) {
            std::string win_name = "Frame " + std::to_string(frame_idx);
            cv::imshow(win_name, frame_cpy);
            cv::waitKey(0);
            cv::destroyWindow(win_name);
        }

        // Increment index.
        frame_idx += frame_step;
        for (int i = 0; i < frame_step; i++) {
            delete bgSubstractor.next_frame();
        }

        if (save_results) {
            cv::imwrite("./res/pipeline_results/dpm-two-persons-1.0-"
                                "threshold/frame" + std::to_string(frame_idx) + ".jpg", frame_cpy);
        }

        frames_results.push_back(frame_cpy.clone());

        // Free the player vector.
        for (size_t i = 0; i < players.size(); i++) {
            delete players[i];
        }
        // Free the frame.
        delete frame;
    }

    size_t results_count = frames_results.size();
    for (size_t i = 0; i < results_count; i++) {
        cv::imshow("Result", frames_results[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
}
