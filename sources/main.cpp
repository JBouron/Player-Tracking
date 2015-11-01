#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../headers/heuristic_feature_extractor.h"
#include "../headers/frame_t.h"
#include "../headers/debug.h"
#include "../headers/bgsubstractor.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/feature_comparator.h"
#include "../headers/dpm_detector.h"
#include "../headers/player_t.h"


#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else

#include <dirent.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <fstream>

#endif


using namespace std;
using namespace cv;

void heuristic_features_extractor_tests() {
    tmd::player_t p;
    p.original_image = (imread(
            "/home/nicolas/Desktop/23102.jpg"));
    tmd::HeuristicFeaturesExtractor d;
    d.extract_features(&p);
    cv::namedWindow("Strips");
/*    for (int i = 0; i < p.features.strips.size(); i ++){
        imshow("Strips", p.features.strips[i]);
        waitKey(0);
    }*/
}

void bgs_demo() {
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
    VideoCapture *capture = new VideoCapture(
            "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/videos/ace_1.mp4");
    tmd::BGSubstractor bgs(capture, 1);
    while (bgs.has_next_frame()) {
        tmd::debug("New iteration");
        tmd::frame_t *frame = bgs.next_frame();
        imshow("Frame", frame->original_frame);
        imshow("FG Mask MOG 2", frame->mask_frame);
        frame->original_frame.release();
        frame->mask_frame.release();
        free(frame);
        cv::waitKey(1);
    }
    tmd::debug("End");
}

void show_body_parts(cv::Mat image, std::vector<cv::Rect> parts) {
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
        cv::rectangle(image, r, color, thickness, line_type, shift);
        cv::imshow("Result", image);
        cv::waitKey(0);
    }
}

std::vector<tmd::player_t *> get_vector() {
    std::vector<tmd::player_t *> v;
    for (int i = 0; i < 3; i++) {
        v.push_back(new tmd::player_t);
        v[i]->original_image = cv::imread(
                "/home/nicolas/Desktop/img" +
                std::to_string(i + 1) +
                ".jpg", CV_LOAD_IMAGE_UNCHANGED);
    }
    return v;

}

void manual_player_comparator_test() {
    tmd::DPMDetector d("/home/nicolas/Documents/EPFL/Projet/Code/Bachelor-Project/res/xmls/person.xml", 4);
    std::vector<tmd::player_t *> v = get_vector();
    Mat data(0,3,CV_32F), labels;
    Mat centers;
    tmd::FeatureComparator comparator(data, 3, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3,
            KMEANS_PP_CENTERS, centers);

    for(int i = 0; i < v.size(); i ++){
        d.extractBodyParts(v[i]);
        comparator.addPlayerFeatures(v[i], 1);
    }

    comparator.runClustering();
    comparator.writeCentersToFile();
    Mat readCenters = comparator.readCentersFromFile();
    comparator.~FeatureComparator();
}

void manual_player_extractor_test() {
    tmd::frame_t frame;
    (frame.original_frame) = (imread(
            "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/img1.jpg"));
    tmd::ManualPlayerExtractor pe = tmd::ManualPlayerExtractor();
    std::vector<tmd::player_t *> v = pe.extract_player_from_frame(&frame);
   // std::vector<tmd::player_t*> v = get_vector();
    /*
       tmd::DPMDetector dpmDetector(
               "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/xmls/person.xml",
               4);

       for (size_t i = 0; i < v.size(); i++) {
           dpmDetector.extractBodyParts(v[i]);
           show_body_parts(v[i]->original_image, v[i]->features.body_parts);
       }*/
}

void test_dpm_class() {
    tmd::DPMDetector d("/home/nicolas/Documents/EPFL/Projet/Code/Bachelor-Project/res/xmls/person.xml", 4);
    std::vector<tmd::player_t *> v = get_vector();


    for (int i = 0; i < 3; i++) {
        d.extractBodyParts(v[i]);
        show_body_parts(v[i]->original_image, v[i]->features.body_parts);
    }

    for (int i = 0; i < 3; i++) {
        delete v[i];
    }
}

int main(int argc, char* argv[]) {
    manual_player_comparator_test();
    return 0;
}

