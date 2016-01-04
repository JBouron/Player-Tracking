#include <iostream>
#include "../headers/data_structures/frame_t.h"
#include "../headers/background_subtractor/bgsubstractor.h"
#include "../headers/features_extraction/dpm.h"
#include "../headers/pipelines/pipeline.h"
#include "../headers/pipelines/multithreaded_pipeline.h"
#include "../headers/tools/training_set_creator.h"
#include "../headers/pipelines/real_time_pipeline.h"
#include "../headers/data_structures/player_t.h"

void show_body_parts(cv::Mat image, tmd::player_t *p);

void extract_player_image(void);

void dpm_feature_extractor_test(void);

void pipeline(void);

void pipeline_class_tests(void);

void dpm_whole_frame(void);

void test_blob_separation(void);

void create_training_set(void);

void memleak_video_capture(void) {
    std::string file = "./res/videos/alone-green-no-ball/ace_0.mp4";
    cv::VideoCapture capture(file);
    int c = 1;
    while (c < 10) {
        capture.set(CV_CAP_PROP_POS_FRAMES, 100 * c);
        c++;
        cv::Mat frame;
        capture.read(frame);
        cv::imshow("Frame", frame);
        cv::waitKey(0);
    }
}

void test_fast_dpm(void) {
    tmd::DPM fastDPM;
    tmd::frame_t blob;
    blob.frame_index = 0;
    blob.camera_index = 0;
    blob.mask_frame = cv::imread(
            "./res/manual_extraction/frame5847_originalimage0.jpg", 0);
    blob.original_frame = cv::imread(
            "./res/manual_extraction/frame5847_originalimage0.jpg");
    double t1 = cv::getTickCount();
    fastDPM.extract_players_and_body_parts(&blob);
    double t2 = cv::getTickCount();
    std::cout << "Time = " << (t2 - t1) / cv::getTickFrequency() << std::endl;
    std::cout << "end" << std::endl;
}

void params_benchmark();

void bgs_benchmark();

int main(int argc, char *argv[]) {
    create_training_set();
    return 0;
    tmd::Config::load_config();
    /*params_benchmark();
    return 0;*/
    tmd::Pipeline *pipeline = new tmd::MultithreadedPipeline("./res/videos/uni-hockey/", 0, 4, 0, 1200, 10);
    tmd::frame_t *frame = pipeline->next_frame();
    //SDL_Window* window = tmd::SDLBinds::create_sdl_window("Frame");
    double t1 = cv::getTickCount();

    std::string folder = "./res/pipeline_results/complete_pipeline/uni/1100frames/";

    std::ofstream outputFile(folder + "output1100.out");

    if (outputFile.is_open()) {

        while (frame != NULL) {
            outputFile << "Frame " << frame->frame_index << std::endl;
            for(tmd::player_t* player : frame->players){
                outputFile << player->team << std::endl;
            }

            std::string frame_index = std::to_string(frame->frame_index);
            std::string file_name = folder + "/frame" + frame_index + ".jpg";
            std::cout << "Save frame " << frame_index << std::endl;
            cv::imwrite(file_name, tmd::draw_player_on_frame(0, frame, true, true, false, false, true));
            tmd::free_frame(frame);
            frame = pipeline->next_frame();
        }
        outputFile.flush();
        outputFile.close();
    }

    delete pipeline;
    double t2 = cv::getTickCount();
    std::cout << "Time = " << (t2 - t1) / cv::getTickFrequency() << std::endl;
    return EXIT_SUCCESS;
}

void params_benchmark() {
    std::string folder = "./res/params_benchmark/";
    tmd::SimplePipeline *pipeline = NULL;
    /*for (tmd::Config::bgs_blob_threshold_count = 1 ;
         tmd::Config::bgs_blob_threshold_count <= 24 ;
         tmd::Config::bgs_blob_threshold_count ++){*/

    for (tmd::Config::dpm_extractor_score_threshold = -5.f;
         tmd::Config::dpm_extractor_score_threshold <= 5.f;
         tmd::Config::dpm_extractor_score_threshold += 0.5f) {

        for (tmd::Config::dpm_extractor_overlapping_threshold = 0.0;
             tmd::Config::dpm_extractor_overlapping_threshold <= 1.0;
             tmd::Config::dpm_extractor_overlapping_threshold += 0.1) {

            pipeline = new tmd::SimplePipeline(
                    "./res/videos/uni-hockey/", 0, 49, 51, 1);
            tmd::frame_t *frame = pipeline->next_frame();
            cv::Mat result = tmd::draw_player_on_frame(0, frame, true,
                                                       true, false,
                                                       false, true);
            std::string file_name = folder + "btc_" + std::to_string
                    (tmd::Config::bgs_blob_threshold_count) + "__dst_" +
                                    std::to_string(tmd::Config::dpm_extractor_score_threshold) +
                                    "__dot_" + std::to_string
                                            (tmd::Config::dpm_extractor_overlapping_threshold) + ".jpg";
            std::cout << "Save : " << file_name << std::endl;
            cv::imwrite(file_name, result);
            tmd::free_frame(frame);
            delete pipeline;
        }
    }
    //}
}

void bgs_benchmark() {
    std::string folder = "./res/bgs_benchmark/";
    tmd::BGSubstractor *bgs;
    tmd::BlobPlayerExtractor *pe = new tmd::BlobPlayerExtractor();

    for (tmd::Config::bgs_blob_buffer_size = 4;
         tmd::Config::bgs_blob_buffer_size <= 6;
         tmd::Config::bgs_blob_buffer_size++) {

        int max = (2 * tmd::Config::bgs_blob_buffer_size + 1) *
                  (2 * tmd::Config::bgs_blob_buffer_size + 1);

        for (tmd::Config::bgs_blob_threshold_count = 1;
             tmd::Config::bgs_blob_threshold_count <= max;
             tmd::Config::bgs_blob_threshold_count++) {

            bgs = new tmd::BGSubstractor("./res/videos/uni-hockey/", 0, 49,
                                         51, 1);

            tmd::frame_t *frame = bgs->next_frame();
            frame->players = pe->extract_player_from_frame(frame);

            cv::Mat result = tmd::draw_player_on_frame(2, frame, false,
                                                       false, false, true,
                                                       false);
            std::string file_name = folder + "bbs_" + std::to_string
                    (tmd::Config::bgs_blob_buffer_size) + "__btc_" +
                                    std::to_string(tmd::Config::bgs_blob_threshold_count) + ".jpg";
            std::cout << "Save : " << file_name << std::endl;
            cv::imwrite(file_name, result);
            tmd::free_frame(frame);
            delete bgs;
        }
    }
}

void create_training_set(void) {
    tmd::Config::load_config();

    tmd::TrainingSetCreator *trainer = new tmd::TrainingSetCreator("./res/videos/uni-hockey/", 0, 0, 1200, 1);
    tmd::frame_t *frame = trainer->next_frame();

    while (frame != NULL) {
        std::string frame_index = std::to_string(frame->frame_index);
        std::cout << "Finished frame " << frame_index << std::endl;

        /*
        if (frame->frame_index != 0 && frame->frame_index % 100 == 0) {
            trainer->write_centers(frame->frame_index);
        }*/

        tmd::free_frame(frame);
        frame = trainer->next_frame();
    }

    tmd::free_frame(frame);
    trainer->write_centers();

    delete trainer;
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

    tmd::DPMDetector dpmDetector;
    dpmDetector.extractBodyParts(player);
    show_body_parts(player->original_image, player);
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