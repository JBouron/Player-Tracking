#include <iostream>
#include "../headers/data_structures/frame_t.h"
#include "../headers/background_subtractor/bgsubstractor.h"
#include "../headers/features_extraction/dpm.h"
#include "../headers/pipelines/pipeline.h"
#include "../headers/pipelines/multithreaded_pipeline.h"
#include "../headers/tools/training_set_creator.h"
#include "../headers/pipelines/real_time_pipeline.h"
#include "../headers/data_structures/cmd_args_t.h"

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
    while (c < 10){
        capture.set(CV_CAP_PROP_POS_FRAMES, 100*c);
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

void show_help();
tmd::cmd_args_t *parse_args(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    tmd::cmd_args_t* args = parse_args(argc, argv);
    if (args == NULL){
        show_help();
        return EXIT_FAILURE;
    }

    tmd::Config::load_config();

    /* The pipeline of the algorithm. */
    tmd::Pipeline *pipeline = NULL;
    SDL_Window *window = NULL;

    if (args->show_results){
        pipeline = new tmd::RealTimePipeline(args->video_folder, args->t,
                                             args->b, args->camera_index,
                                             args->s, args->e);
    }
    else{
        if (args->t == 1){
            pipeline = new tmd::SimplePipeline(args->video_folder,
                                               args->camera_index, args->s,
                                               args->e, args->j);
        }
        else if (args->t > 1){
            pipeline = new tmd::MultithreadedPipeline(args->video_folder,
                                                      args->camera_index,
                                                      args->t, args->s,
                                                      args->e, args->j);
        }
        else{
            std::cout << "Error, invalid thread count : " << args->t <<
                    std::endl;
            return EXIT_FAILURE;
        }
    }
    /*cv::VideoCapture capture(args->video_folder + "/ace_" + std::to_string
                                                (args->camera_index) +".mp4");
    cv::Size S = cv::Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),    //
            // Acquire input size
                  (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter videoWriter(args->save_folder + "/ace_0.mp4",
                                static_cast<int> (capture.get
                (CV_CAP_PROP_FOURCC)), capture.get(CV_CAP_PROP_FPS), S, true);*/
    tmd::frame_t *frame = pipeline->next_frame();
    std::cout << "Begin" << std::endl;
    while (frame != NULL){
        cv::Mat result = tmd::draw_player_on_frame(0, frame, true,
                                                   args->show_torsos, false,
                                                   false, true);
        if (args->show_results){
            // TODO
        }

        if (args->save_results){
            std::string file_name = args->save_folder + "/frame" +
                    std::to_string(frame->frame_index) + ".jpg";
            std::cout << "Save frame " << frame->frame_index << std::endl;
            cv::imwrite(file_name, result);
            //videoWriter.write(result);
        }
        free_frame(frame);
        frame = pipeline->next_frame();
    }

    std::cout << "Done" << std::endl;
    return EXIT_SUCCESS;

    /**/
    /*tmd::Pipeline *pipeline = new tmd::RealTimePipeline(
                        "./res/videos/alone-red-no-ball/", 4, 2, 0, 0, 1200);
    tmd::frame_t *frame = pipeline->next_frame();
    SDL_Window* window = tmd::SDLBinds::create_sdl_window("Frame");
    double t1 = cv::getTickCount();
    int count = 0;
    int max_frames = -1;
    std::string folder = "./res/pipeline_results/complete_pipeline/alone-red"
            "-no-ball/";

    while (frame != NULL) {
        std::string frame_index = std::to_string(count);
        std::string file_name = folder + "/frame" + frame_index + ".jpg";
        std::cout << "Save frame " << frame_index << std::endl;
        tmd::SDLBinds::imshow(window, tmd::draw_player_on_frame(0, frame,
                                                                true));
        delete frame;
        frame = pipeline->next_frame();
        count++;
        if (count == max_frames) {
            break;
        }
    }

    delete pipeline;
    double t2 = cv::getTickCount();
    std::cout << "Time = " << (t2 - t1) / cv::getTickFrequency() << std::endl;
    return EXIT_SUCCESS;*/
}

void show_help(){
    //tmd ./videos camera-index --show-result -s 0 -e 120 -j 10 -t 4 --save-result --show-torsos -b 5
    std::cout << "###############################################" << std::endl;
    std::cout << "#             Bachelor project                #" << std::endl;
    std::cout << "###############################################" << std::endl;

    std::cout << "              Help                             " << std::endl;
    std::cout << "The first argument should be the folder containing the "
                         "videos."  << std::endl;
    std::cout << "The second argument is the camera index on which the "
                         "algorithm will run. This index is between 0 and 7 "
                         "included." << std::endl;
    std::cout << "Then any of the following arguments can be added in any "
                         "order, if they are not specified, default values "
                         "will be used" << std::endl;
    std::cout << "--show-results Show result at every frame in a popup window"
                         "." << std::endl;
    std::cout << "--save-results path Create a video file and save it to the "
                         "specified path. (The name is the same as the input "
                         "video)" << std::endl;
    std::cout << "--show-torsos Show the torso boxes on the resulting frames"
                         "." << std::endl;
    std::cout << "-s number Set the starting frame to number. (default : 0)" <<
            std::endl;
    std::cout << "-e number Set the ending frame to number. (default : last "
                         "frame)" << std::endl;
    std::cout << "-j size Set the step size. (default : 1)" << std::endl;
    std::cout << "-t count Set the number of threads to use. (default : 1)"
    << std::endl;
    std::cout << "-b rate Set the refresh rate of the player boxes. (default "
                         ": every frames)" << std::endl;

}

tmd::cmd_args_t *parse_args(int argc, char *argv[]){
    tmd::cmd_args_t *args = new tmd::cmd_args_t;
    if (argc < 3){
        std::cout << "Error, expected at least 2 arguments." << std::endl;
        delete args;
        return NULL;
    }

    args->video_folder = argv[1];
    args->camera_index = static_cast<int> (strtol(argv[2], NULL, 10));

    for (int i = 3 ; i < argc ; i ++){
        if (!strcmp(argv[i], "--show-results")){
            args->show_results = true;
        }
        else if (!strcmp(argv[i], "--save-results")){
            args->save_results = true;
            i ++;
            if (i == argc){
                std::cout << "Error, expected save folder." << std::endl;
                return NULL;
            }
            else args->save_folder = argv[i];
        }
        else if (!strcmp(argv[i], "--show-torsos")){
            args->show_torsos = true;
        }
        else if (!strcmp(argv[i], "-s")){
            if (i == argc - 1){
                std::cout << "Error, expected starting frame." << std::endl;
                return NULL;
            }
            else{
                i ++;
                args->s = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-e")){
            if (i == argc - 1){
                std::cout << "Error, expected ending frame." << std::endl;
                return NULL;
            }
            else{
                i ++;
                args->e = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-j")){
            if (i == argc - 1){
                std::cout << "Error, expected step size." << std::endl;
                return NULL;
            }
            else{
                i ++;
                args->j = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-t")){
            if (i == argc - 1){
                std::cout << "Error, expected thread count." << std::endl;
                return NULL;
            }
            else{
                i ++;
                args->t = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-b")){
            if (i == argc - 1){
                std::cout << "Error, expected box refresh rate." << std::endl;
                return NULL;
            }
            else{
                i ++;
                args->b = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else{
            std::cout << "Error, unknown argument : " << argv[i] <<
                    std::endl;
            delete args;
            return NULL;
        }
    }
    return args;
}

void create_training_set(void) {
    tmd::Config::load_config();

    tmd::TrainingSetCreator *trainer = new tmd::TrainingSetCreator("./res/videos/uni-hockey/", 0,
                                                                   "./res/xmls/person.xml", 0, 300, 1);
    tmd::frame_t *frame = trainer->next_frame();

    int count = 0;
    int max_frames = 10;

    while (frame != NULL) {
        std::string frame_index = std::to_string(count);
        std::cout << "Finished frame " << frame_index << std::endl;
        tmd::free_frame(frame);
        frame = trainer->next_frame();
        count++;
        if (count == max_frames) {
            break;
        }
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