#include <iostream>
#include "../headers/frame_t.h"
#include "../headers/test_cases/test_suite.h"
#include "../headers/demo/demo.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/calibration_tool.h"

void extract_player_image(void);

int main(int argc, char *argv[]) {
    cv::VideoCapture* video = new cv::VideoCapture("/home/nicolas/Documents/EPFL/Projet/Bachelor-Project/res/videos/alone-green-no-ball/ace_04.mp4");
    tmd::BGSubstractor s(video, 1);

    //(tmd::run_demo_pipeline();
    return EXIT_SUCCESS;
}

void extract_player_image(void){
    cv::VideoCapture capt("./misc/ace_0.mp4");
    tmd::BGSubstractor bgs(&capt, 0);

    int keyboard = 0;
    cv::namedWindow("Extraction");
    tmd::frame_t* frame;
    while (keyboard != 27){
        keyboard = cv::waitKey(15);
        frame = bgs.next_frame();
        cv::imshow("Extraction", frame->original_frame);
        if (keyboard != 27){
            delete frame;
        }
    }

    tmd::ManualPlayerExtractor mp;
    mp.extract_player_from_frame(frame);
}

