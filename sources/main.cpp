#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../headers/frame_t.h"


int main(int argc, char *argv[]) {
    cv::imshow("Image", cv::imread("./misc/images/hue360.jpg"));
    cv::waitKey(0);
    //tmd::features_extractor_tests("./misc/images/hue360.jpg");
    return EXIT_SUCCESS;
}

