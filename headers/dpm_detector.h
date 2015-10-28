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
        DPMDetector(std::string model_file, float overlap_threshold, int numthread);
        ~DPMDetector();

        void extractTorso(tmd::player_t* player);
        void testOnImage(IplImage* image);

    private:
        /* Private methods, those are custom redefinitions of the ones coming
         * from openCV.
         */
        static int CustomEstimateBoxes(CvPoint *points, int *levels,
                                       int kPoints,
                                       int sizeX, int sizeY,
                                       CvPoint **oppositePoints);
        /*int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                             const int *kPartFilters,
                             unsigned int *maxXBorder, unsigned int *maxYBorder);*/
        int CustomShowPartFilterBoxes(IplImage *image,
                                      const CvLSVMFilterObject **filters,
                                      int n, CvPoint **partsDisplacement,
                                      int *levels, int kPoints,
                                      CvScalar color, int thickness,
                                      int line_type, int shift, float *scores);
        int CustomSearchObjectThresholdSomeComponents(IplImage *image,
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
        /*int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                                  const CvPoint *oppositePoints, const float *score,
                                  float overlapThreshold,
                                  int *numBoxesOut, CvPoint **pointsOut,
                                  CvPoint **oppositePointsOut, float **scoreOut);*/
        CvSeq* CustomcvLatentSvmDetectObjects(IplImage* image,
                                              CvLatentSvmDetector* detector,
                                              CvMemStorage* storage,
                                              float overlap_threshold, int numThreads);

        CvLatentSvmDetector* m_detector;
        CvMemStorage* m_memory_storage;
        float m_overlap_threshold;
        int m_numthread;
    };
}

#endif //BACHELOR_PROJECT_DPM_DETECTOR_H_H
