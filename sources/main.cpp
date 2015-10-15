//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C++
#include <iostream>
#include <sstream>

#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "../headers/bgsubstractor.h"
#include "../headers/frame_t.h"
#include "../headers/debug.h"
#include "../headers/calibration_tool.h"

using namespace cv;
using namespace std;

void calibration_demo(){
    tmd::CalibrationTool cal("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/videos/");
    cal.calibrate();
    float** params = cal.retrieve_params();
    for (int i = 0; i < 8; i ++){
        tmd::debug("camera " + std::to_string(i) + "   Th = " + std::to_string(params[i][0]) + "   HS = " + std::to_string(params[i][1]) + "   lr = " + std::to_string(params[i][2]));
        free(params[i]);
    }
    free(params);
}

void bgs_demo(){
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
    VideoCapture* capture = new VideoCapture("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/videos/ace_1.mp4");
    tmd::BGSubstractor bgs(capture, 1);
    while (bgs.has_next_frame()){
        tmd::debug("New iteration");
        tmd::frame_t* frame = bgs.next_frame();
        imshow("Frame", *frame->original_frame);
        imshow("FG Mask MOG 2", *frame->mask_frame);
        frame->original_frame->release();
        frame->mask_frame->release();
        free(frame->original_frame);
        free(frame->mask_frame);
        free(frame);
        cv::waitKey(1);
    }
    tmd::debug("End");
}

int main(int argc, char* argv[]) {
    bgs_demo();
    destroyAllWindows();
    return EXIT_SUCCESS;
}