#ifndef BACHELOR_PROJECT_FAST_DPM_H
#define BACHELOR_PROJECT_FAST_DPM_H

#include "frame_t.h"
#include "../headers/config.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "../headers/openCV/_latentsvm.h"
#include "../headers/openCV/_lsvm_matching.h"
#include "sdl_binds/sdl_binds.h"

namespace tmd{
    class FastDPM{
    public:
        FastDPM();

        std::vector<tmd::player_t*> extract_players_and_body_parts(cv::Mat
                                                                   blob);

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

        int showPartFilterBoxes(IplImage *image,
                                const CvLSVMFilterObject **filters,
                                int n, CvPoint **partsDisplacement,
                                int *levels, int kPoints,
                                CvScalar color, int thickness,
                                int line_type, int shift);

        CvScalar m_color;
        int m_thickness;
        int m_line_type;
        int m_shift;
        cv::Mat m_image;
        CvLatentSvmDetector *m_detector;

        std::vector<std::vector<cv::Rect> > m_parts;

    };
}

#endif //BACHELOR_PROJECT_FAST_DPM_H
