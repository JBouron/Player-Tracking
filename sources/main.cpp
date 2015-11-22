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

void extract_player_image(void);
void dpm_extractor(void);
void pipeline(void);

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

void pipeline(void){
    std::string video_path = "./res/videos/alone-green-no-ball/ace_0.mp4";
    cv::VideoCapture input_video(video_path);
    tmd::BGSubstractor bgs(&input_video, 0);
    tmd::DPMPlayerExtractor dpmPlayerExtractor("./res/xmls/person.xml");
    tmd::FeaturesExtractor featuresExtractor("./res/xmls/person.xml");


    /** Ca ça vient de la démo : **/
    int clusterCols = 180;
    int clusterCount = 2;
    cv::Mat data, labels(1, clusterCols, CV_32F);
    std::cout << data.cols << std::endl;
    cv::Mat clusterCenters = cv::Mat(clusterCount, clusterCols, CV_32F);
    cv::TermCriteria termCriteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0);
    int attempts = 3;
    int flags = cv::KMEANS_PP_CENTERS;
    tmd::FeatureComparator comparator(data, clusterCount, labels, termCriteria, attempts, flags, clusterCenters);
}