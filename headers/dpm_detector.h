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

        void extractTorso(tmd::player_t* player);
        void testOnImage(IplImage* image);

    private:
        /* private structure holding part boxes infos */
        typedef struct{
            std::vector<cv::Rect> boxes;
            int kPartFilters; // Number of part filters.
            CvPoint** part_displacement; // Position of the parts.
            int* levels; // levels of each parts.
            int kPoint; // Number of points.
            float* scores; // scores for each parts.
        }part_boxes_t;

        /* Private methods, those are custom redefinitions of the ones coming
         * from openCV.
         */
        static int CustomEstimateBoxes(CvPoint *points, int *levels,
                                       int kPoints,
                                       int sizeX, int sizeY,
                                       CvPoint **oppositePoints);

        int detectBestPartBoxes(part_boxes_t* parts, IplImage *image,
                                      const CvLSVMFilterObject **filters,
                                      int n, CvPoint **partsDisplacement,
                                      int *levels, int kPoints, float *scores);


        part_boxes_t* getPartBoxesForImage(IplImage* image);

        int fillPartStruct(part_boxes_t* parts, IplImage *image,
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
