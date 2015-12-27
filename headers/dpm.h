#ifndef BACHELOR_PROJECT_FAST_DPM_H
#define BACHELOR_PROJECT_FAST_DPM_H

#include "frame_t.h"
#include "../headers/config.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "../headers/openCV/_lsvm_matching.h"
#include "../headers/openCV/_latentsvm.h"
#include "../headers/openCV/_lsvm_distancetransform.h"
#include "../headers/openCV/_lsvm_error.h"
#include "../headers/openCV/_lsvm_fft.h"
#include "../headers/openCV/_lsvm_routine.h"
#include "../headers/openCV/_lsvm_types.h"
#include "sdl_binds/sdl_binds.h"

namespace tmd{
    typedef std::tuple<cv::Rect, std::vector<cv::Rect>, float> detection;

    class DPM {
    public:
        DPM();

        std::vector<tmd::player_t*> extract_players_and_body_parts
                (tmd::frame_t* frame);

    private:
        /* cvLatentSvmDetectObjects */
        CvSeq* cvLatentSvmDetectObjects(IplImage* image,
                                        CvLatentSvmDetector* detector,
                                        CvMemStorage* storage,
                                        float overlap_threshold, int
                                        numThreads);

        /* searchObjectThresholdSomeComponents */
        int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H,
                                                const CvLSVMFilterObject **filters,
                                                int kComponents, const int *kPartFilters,
                                                const float *b, float scoreThreshold,
                                                CvPoint **points, CvPoint **oppPoints,
                                                float **score, int *kPoints, int numThreads);

        /* EstimateBoxes */
        int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                          int sizeX, int sizeY, CvPoint **oppositePoints);

        int searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                                  const CvLSVMFilterObject **all_F, int n,
                                  float b,
                                  int maxXBorder, int maxYBorder,
                                  float scoreThreshold,
                                  CvPoint **points, int **levels, int *kPoints,
                                  float **score, CvPoint ***partsDisplacement,
                                  int numThreads);

        int convertPoints(int /*countLevel*/, int lambda,
                          int initialImageLevel,
                          CvPoint *points, int *levels,
                          CvPoint **partsDisplacement, int kPoints, int n,
                          int maxXBorder,
                          int maxYBorder);

        int getOppositePoint(CvPoint point,
                             int sizeX, int sizeY,
                             float step, int degree,
                             CvPoint *oppositePoint);

        int thresholdFunctionalScore(const CvLSVMFilterObject **all_F,
                                     int n,
                                     const CvLSVMFeaturePyramid *H,
                                     float b,
                                     int maxXBorder, int maxYBorder,
                                     float scoreThreshold,
                                     float **score,
                                     CvPoint **points, int **levels, int *kPoints,
                                     CvPoint ***partsDisplacement);


        int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                                  const CvPoint *oppositePoints, const float *score,
                                  float overlapThreshold,
                                  int *numBoxesOut, CvPoint **pointsOut,
                                  CvPoint **oppositePointsOut, float
                                  **scoreOut);

        std::vector<cv::Rect> get_parts_rect_for_point(const
                                            CvLSVMFilterObject **filters,
                                            int n, CvPoint
                                            *partsDisplacement, int
                                            level);

        int clippingBoxesUpperRightCorner(int width, int height,
                                                   CvPoint *points, int
                                                   kPoints);
        int clippingBoxesLowerLeftCorner(int width, int height,
                                                  CvPoint *points, int kPoints);

        void clamp_detections();
        void extractTorsoForPlayer(player_t *player);

        CvLatentSvmDetector *m_detector;

        std::vector<tmd::detection> m_detections; // box, parts, score.

    };
}

#endif //BACHELOR_PROJECT_FAST_DPM_H
