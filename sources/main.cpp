#include <iostream>
#include "../headers/frame_t.h"
#include "../headers/test_cases/test_suite.h"
#include "../headers/demo/demo.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/calibration_tool.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/player_t.h"
#include "../headers/features_extractor.h"
#include "../headers/feature_comparator.h"
#include "../headers/features_t.h"

void extract_player_image(void);

void dpm_extractor(void);

void dpm_crop_test(void);
void dpm_feature_extractor_test(void);

void pipeline(void);

int main(int argc, char *argv[]) {
    pipeline();
    return EXIT_SUCCESS;
}

void extract_player_image(void) {
    cv::VideoCapture capt("./misc/ace_0.mp4");
    tmd::BGSubstractor bgs(&capt, 0);
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

void dpm_extractor(void) {
    tmd::DPMPlayerExtractor d("./res/xmls/person.xml");

    int image_count = 5;
    for (int i = 0; i < image_count ; i ++){
        std::cout << "player " << i << std::endl;
        tmd::frame_t* frame = new tmd::frame_t;
        frame->original_frame = cv::imread(
                "./res/pipeline_results/player_extraction/player19" + std::to_string
                        (i) + "0-0-resize.jpg");
        frame->mask_frame = cv::imread(
                "./res/pipeline_results/player_extraction/player19" + std::to_string
                        (i) + "0-0-resize.jpg");

        std::vector<tmd::player_t*> res = d.extract_player_from_frame(frame);
        std::cout << res.size() << " players detected." << std::endl;

        for (int i = 0 ; i < res.size() ; i ++){
            delete res[i];
        }
        delete frame;
    }
}


void show_body_parts(cv::Mat image, tmd::player_t* p) {
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
    for (int i = 0; i < parts.size(); i++) {
        CvRect r;
        r.x = parts[i].x;
        r.y = parts[i].y;
        r.width = parts[i].width;
        r.height = parts[i].height;
        cv::rectangle(image, r, color, thickness, line_type, shift);
    }
    cv::imshow("Body parts", image);
    cv::waitKey(0);
}

void pipeline(void) {
    std::string video_path = "./res/videos/alone-green-no-ball/ace_0.mp4";
    cv::VideoCapture input_video(video_path);
    tmd::BGSubstractor bgs(&input_video, 0);
    tmd::DPMPlayerExtractor dpmPlayerExtractor("./res/xmls/person.xml");
    tmd::FeaturesExtractor featuresExtractor("./res/xmls/person.xml");

    // TODO : Cluster.
    int fidx = 290;

    bgs.jump_to_frame(290);
    while (true) {
        std::cout << "fidx = " << fidx << std::endl;
        for (int i = 0; i < 10; i++) {
            delete (bgs.next_frame());
        }
        fidx += 10;
        tmd::frame_t *frame = bgs.next_frame();
        std::vector<tmd::player_t *> players = dpmPlayerExtractor
                .extract_player_from_frame(frame);

        std::cout << players.size() << " players detected." << std::endl;

        CvScalar color;
        color.val[0] = 255;
        color.val[1] = 255;
        color.val[2] = 0;
        color.val[3] = 255;

        CvScalar torso;
        torso.val[0] = 255;
        torso.val[1] = 0;
        torso.val[2] = 255;
        torso.val[3] = 255;

        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        cv::Mat fcpy = frame->original_frame.clone();
        for (int i = 0; i < players.size(); i++) {
            tmd::player_t *player = players[i];
            featuresExtractor.extractFeatures(players[i]);
            CvRect r;
            r.x = player->features.torso_pos.x + player->pos_frame.x;
            r.y = player->features.torso_pos.y + player->pos_frame.y;
            r.width = player->features.torso_pos.width;
            r.height = player->features.torso_pos.height;
            show_body_parts(fcpy, player);
            cv::rectangle(fcpy, r, torso, thickness,
                           line_type,
                          shift);
            cv::rectangle(fcpy, players[i]->pos_frame, color,
                          thickness, line_type, shift);
            cv::imwrite("./res/pipeline_results/player_extraction/player" +
                                std::to_string(fidx) + "-" + std::to_string
                 (i) + ".jpg", frame->original_frame(player->pos_frame));
        }

        std::string file_name = "./res/pipeline_results/torsodetect" +
                std::to_string
                (fidx) + ".jpg";
        cv::imwrite(file_name, fcpy);
        std::cout << "Writing to file : " << file_name << std::endl;
        /*cv::imshow("Frame", fcpy);
        cv::waitKey(0);*/
        delete (frame);
    }
}

void dpm_feature_extractor_test(void){
    tmd::FeaturesExtractor fe("./res/xmls/person.xml");

    tmd::player_t* player = new tmd::player_t;
    player->original_image =
        cv::imread("./res/pipeline_results/player_extraction/player1910-0.jpg");

    player->mask_image =
            cv::imread("./res/pipeline_results/player_extraction/player1910-0"
                               ".jpg");

    fe.extractFeatures(player);

    show_body_parts(player->original_image, player);
    delete player;
}

void dpm_crop_test(void){

}