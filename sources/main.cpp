#include <iostream>
#include "../headers/data_structures/frame_t.h"
#include "../headers/background_subtractor/bgsubstractor.h"
#include "../headers/features_extraction/dpm.h"
#include "../headers/pipelines/pipeline.h"
#include "../headers/pipelines/multithreaded_pipeline.h"
#include "../headers/tools/training_set_creator.h"
#include "../headers/pipelines/approximative_pipeline.h"
#include "../headers/data_structures/cmd_args_t.h"

tmd::cmd_args_t *parse_args(int argc, char *argv[]);
void run_test();
void create_training_set(std::string video_folder,
             int camera_index, int start_frame, int end_frame, int step_size);

int main(int argc, char *argv[]) {
    tmd::cmd_args_t *args = parse_args(argc, argv);
    if (args == NULL) {
        return EXIT_FAILURE;
    }

    if (args->test_run){
        run_test();
        return EXIT_SUCCESS;
    }

    // If the user forgot the '/' ...
    if (args->video_folder[args->video_folder.size()-1] != '/'){
        args->video_folder += '/';
    }

    if (args->training_set_creator){
        create_training_set(args->video_folder,
                            args->camera_index,
                            args->s, args->e, args->j);
        return EXIT_SUCCESS;
    }

    tmd::Config::load_config();

    /* The pipeline of the algorithm. */
    tmd::Pipeline *pipeline = NULL;
    SDL_Window *window = NULL;
    cv::VideoWriter *writer = NULL;
    bool use_approximate_pipeline;

    if (args->b > 1) {
        pipeline = new tmd::ApproximativePipeline(args->video_folder,
                                                  args->camera_index, args->t,
                                                  args->s, args->e, args->b);
        use_approximate_pipeline = true;
    }
    else {
        use_approximate_pipeline = false;
        if (args->t == 1) {
            pipeline = new tmd::SimplePipeline(args->video_folder,
                                               args->camera_index, args->s,
                                               args->e, args->j);
        }
        else if (args->t > 1) {
            pipeline = new tmd::MultithreadedPipeline(args->video_folder,
                                                      args->camera_index,
                                                      args->t, args->s,
                                                      args->e, args->j);
        }
        else {
            std::cout << "Error, invalid thread count : " << args->t <<
            std::endl;
            return EXIT_FAILURE;
        }
    }

    if (tmd::Config::show_results) {
        window = tmd::SDLBinds::create_sdl_window("TMD");
    }

    if (tmd::Config::save_results){
        cv::VideoCapture original_video(args->video_folder + "/ace_" +
                            std::to_string(args->camera_index) + ".mp4");
        double fps = original_video.get(CV_CAP_PROP_FPS);

        cv::Size size((int)(original_video.get(CV_CAP_PROP_FRAME_WIDTH)),
                      (int)(original_video.get(CV_CAP_PROP_FRAME_HEIGHT)));

        std::string video_path = "result.avi";

        double writer_fps;
        if (use_approximate_pipeline){
            writer_fps = fps;
        }
        else{
            writer_fps = fps / args->j;
        }

        writer = new cv::VideoWriter(video_path, CV_FOURCC('M', 'P', '4', 'V')
                , writer_fps, size, true);
    }

    tmd::frame_t *frame = pipeline->next_frame();

    std::cout << "Begin" << std::endl;
    double t1 = cv::getTickCount();
    while (frame != NULL) {
        cv::Mat result = tmd::draw_player_on_frame(0, frame);

        if (tmd::Config::show_results) {
            tmd::SDLBinds::imshow(window, result);
        }

        if (tmd::Config::save_results) {
            std::cout << "Write frame " << frame->frame_index << std::endl;
            writer->write(result);
        }

        if (tmd::Config::save_all_frames){
            std::string file_name = "./frames/frame" + std::to_string(
                                                   frame->frame_index) + ".jpg";
            std::cout << "Save frame " << frame->frame_index << std::endl;
            cv::imwrite(file_name, result);
        }

        if (!use_approximate_pipeline) {
            free_frame(frame);
        }
        std::cout << "Frame " << frame->frame_index << " done" << std::endl;
        frame = pipeline->next_frame();
    }
    double t2 = cv::getTickCount();

    std::cout << "Done" << std::endl;
    std::cout << "Time = " << (t2 - t1) / cv::getTickFrequency() << std::endl;

    if (tmd::Config::show_results) {
        tmd::SDLBinds::destroy_sdl_window(window);
        tmd::SDLBinds::quit_sdl();
    }

    if (tmd::Config::save_results){
        delete writer;
    }

    delete args;
    delete pipeline;
    return EXIT_SUCCESS;
}

tmd::cmd_args_t *parse_args(int argc, char *argv[]) {
    tmd::cmd_args_t *args = new tmd::cmd_args_t;
    if (argc < 2) {
        std::cout << "Error, expected at least 2 arguments." << std::endl;
        delete args;
        return NULL;
    }
    else if (!strcmp(argv[1], "--test")){
        args->test_run = true;
        return args;
    }

    if (argc < 3) {
        std::cout << "Error, expected at least 2 arguments." << std::endl;
        delete args;
        return NULL;
    }

    args->video_folder = argv[1];
    args->camera_index = static_cast<int> (strtol(argv[2], NULL, 10));

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--show")) {
            tmd::Config::show_results = true;
        }
        else if (!strcmp(argv[i], "--train")) {
            args->training_set_creator = true;
        }
        else if (!strcmp(argv[i], "-s")) {
            if (i == argc - 1) {
                std::cout << "Error, expected starting frame." << std::endl;
                return NULL;
            }
            else {
                i++;
                args->s = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-e")) {
            if (i == argc - 1) {
                std::cout << "Error, expected ending frame." << std::endl;
                return NULL;
            }
            else {
                i++;
                args->e = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-j")) {
            if (i == argc - 1) {
                std::cout << "Error, expected step size." << std::endl;
                return NULL;
            }
            else {
                i++;
                args->j = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-t")) {
            if (i == argc - 1) {
                std::cout << "Error, expected thread count." << std::endl;
                return NULL;
            }
            else {
                i++;
                args->t = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else if (!strcmp(argv[i], "-b")) {
            if (i == argc - 1) {
                std::cout << "Error, expected box refresh rate." << std::endl;
                return NULL;
            }
            else {
                i++;
                args->b = static_cast<int>(strtol(argv[i], NULL, 10));
            }
        }
        else {
            std::cout << "Error, unknown argument : " << argv[i] <<
            std::endl;
            delete args;
            return NULL;
        }
    }
    return args;
}

void create_training_set(std::string video_folder,
             int camera_index, int start_frame, int end_frame, int step_size) {
    tmd::Config::load_config();

    tmd::TrainingSetCreator *trainer =
            new tmd::TrainingSetCreator(video_folder, camera_index, start_frame,
                                                        end_frame, step_size);
    tmd::frame_t *frame = trainer->next_frame();

    while (frame != NULL) {
        std::string frame_index = std::to_string(frame->frame_index);
        std::cout << "Finished frame " << frame_index << std::endl;

        tmd::free_frame(frame);
        frame = trainer->next_frame();
    }

    tmd::free_frame(frame);
    trainer->write_centers();

    delete trainer;
}

void run_test(){
    tmd::SimplePipeline pipeline("./test/", 0, 0,
                                            std::numeric_limits<int>::max(), 1);
    tmd::frame_t *frame = pipeline.next_frame();
    while (frame != NULL) {

        std::cout << "Frame " << frame->frame_index << std::endl;
        for (tmd::player_t *player : frame->players) {
            std::cout << player->team << std::endl;
        }
        free_frame(frame);
        frame = pipeline.next_frame();
    }
}