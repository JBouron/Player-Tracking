#ifndef BACHELOR_PROJECT_DPM_DETECTOR_H_H
#define BACHELOR_PROJECT_DPM_DETECTOR_H_H

#include <string>
#include <cmath>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "../headers/openCV/_latentsvm.h"
#include "../headers/openCV/_lsvm_matching.h"
#include "player_t.h"

/* Number of threads used during the DPM extraction. */
#define TMD_DPM_DETECTOR_NUMTHREADS 4

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
        /**
         * Constructor of the class taking the file path containing the xml
         * file for the person model.
         */
        DPMDetector(std::string model_file);
        ~DPMDetector();

        /**
         * Extract the body parts of the player given in parameter.
         * Store the body part in the feature struct of the player.
         */
        void extractBodyParts(tmd::player_t *player);

    private:
        /* Private methods, those are custom redefinitions of the ones coming
         * from openCV.
         */

        /**
         * Estimate the position of each filter.
         * This method is exactly the same as the estimateBoxes function
         * from openCV, but is here because estimateBoxes has no prototype
         * thus it was not possible to reuse it.
         */
        static int customEstimateBoxes(CvPoint *points, int *levels,
                                       int kPoints,
                                       int sizeX, int sizeY,
                                       CvPoint **oppositePoints);

        /**
         * Detect the best part boxes based on their level and their score.
         * It is a modified version of showPartBoxes from openCV.
         */
        int detectBestPartBoxes(std::vector<cv::Rect>& parts, IplImage *image,
                                      const CvLSVMFilterObject **filters,
                                      int n, CvPoint **partsDisplacement,
                                      int *levels, int kPoints, float *scores);

        /**
         * Extract the part boxes for a given image.
         * Returns a vector containing the position of each part.
         */
        std::vector<cv::Rect> getPartBoxesForImage(IplImage* image);

        /**
         * prepare the detector to detect the person and place the filters/parts
         * boxes accordingly.
         * This is a custom version of the CvLatentSvmDetect function, where
         * the 'detection' part (drawing candidates and sort them) has been
         * removed.
         */
        int preparePartDetection(std::vector<cv::Rect> &parts, IplImage *image,
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

        /**
         * Fill the torso attribute of the player's feature.
         * The torso is the 'mean' of the 2 box filter of the dpm detector.
         */
        void extractTorsoForPlayer(player_t* player);

        /**
         * Make sure that the boundaries of each part boxes are not 'outside'
         * the player image. Resize them if needed.
         */
        void clipBoxes(player_t* player);

        void shrinkBox(player_t* player);

        /* Instance of the LatentSvmDetector used to extract parts. */
        CvLatentSvmDetector* m_detector;
        /* Number of threads used during the process. */
        int m_numthreads;
    };
}

#endif //BACHELOR_PROJECT_DPM_DETECTOR_H_H
