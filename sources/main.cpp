#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../headers/frame_t.h"
#include "../headers/debug.h"
#include "../headers/bgsubstractor.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/feature_comparator.h"
#include "../headers/dpm_detector.h"
#include "../headers/player_t.h"
#include "../headers/features_extractor.h"
#include "../headers/features_t.h"
#include "../tests/features_extractor_tests.h"


#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else

#include <dirent.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <fstream>

#endif


using namespace std;
using namespace cv;



int main(int argc, char *argv[]) {
    tmd::features_extractor_tests();
    return EXIT_SUCCESS;
}

