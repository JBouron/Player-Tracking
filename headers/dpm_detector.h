#ifndef BACHELOR_PROJECT_DPM_DETECTOR_H_H
#define BACHELOR_PROJECT_DPM_DETECTOR_H_H

#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include "../../../../../openCV-2.4.11/opencv-2.4.11/modules/objdetect/src/_latentsvm.h"
#include "../../../../../openCV-2.4.11/opencv-2.4.11/modules/objdetect/src/_lsvm_matching.h"
#include "player_t.h"

namespace tmd{
    /**
     * Class representing a DPM detector using the LatentSVM algorithm.
     * The code of this class is a mix of openCV code for LatentSVMDetector and
     * our code in order to add some functionality.
     *
     * This class is a wrapper of a custom CvLatentSVMDetector from openCV.
     */

    class DPMDetector{
    public :
        DPMDetector(std::string model_file);
        ~DPMDetector();

        void extractBodyParts(tmd::player_t *player);

    private:
        /* Private methods, those are custom redefinitions of the ones coming
         * from openCV.
         */
        static int CustomEstimateBoxes(CvPoint *points, int *levels,
                                       int kPoints,
                                       int sizeX, int sizeY,
                                       CvPoint **oppositePoints);

        int detectBestPartBoxes(std::vector<cv::Rect>& parts, IplImage *image,
                                      const CvLSVMFilterObject **filters,
                                      int n, CvPoint **partsDisplacement,
                                      int *levels, int kPoints, float *scores);


        std::vector<cv::Rect> getPartBoxesForImage(IplImage* image);

        int fillPartStruct(std::vector<cv::Rect>& parts, IplImage *image,
                           const CvLSVMFeaturePyramid *H,
                           const CvLSVMFilterObject **filters,
                           int kComponents,
                           const int *kPartFilters,
                           const float *b,
                           float scoreThreshold,
                           CvPoint **points,
                           CvPoint **oppPoints,
                           float **score,
                           int *kPoints,
                           int numThreads);


        CvLatentSvmDetector* m_detector;
        int m_numthread;
    };
}

#endif //BACHELOR_PROJECT_DPM_DETECTOR_H_H
