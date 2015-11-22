#include <iostream>
#include "../headers/frame_t.h"
#include "../headers/test_cases/test_suite.h"
#include "../headers/demo/demo.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/calibration_tool.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/player_t.h"

void extract_player_image(void);
void dpm_extractor(void);

int main(int argc, char *argv[]) {
    dpm_extractor();
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

void dpm_extractor(void){
    tmd::DPMPlayerExtractor d("./res/xmls/person.xml");
    tmd::frame_t* frame = new tmd::frame_t;
    frame->original_frame = cv::imread("./res/images/uni2.jpg");
    //frame->mask_frame = cv::imread("./res/demo/screenshot4.png");

    std::vector<tmd::player_t*> res = d.extract_player_from_frame(frame);

    std::cout << "Total detections = " << res.size() << std::endl;
    for (size_t i = 0 ; i < res.size() ; i ++){
        cv::imshow("Result", res[i]->original_image);
        cv::waitKey(0);
    }
}