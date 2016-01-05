#ifndef BACHELOR_PROJECT_FAST_DPM_H
#define BACHELOR_PROJECT_FAST_DPM_H

#include "../data_structures/frame_t.h"
#include "../misc/config.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "../openCV/_lsvm_matching.h"
#include "../openCV/_latentsvm.h"
#include "../openCV/_lsvm_distancetransform.h"
#include "../openCV/_lsvm_error.h"
#include "../openCV/_lsvm_fft.h"
#include "../openCV/_lsvm_routine.h"
#include "../openCV/_lsvm_types.h"
#include "../sdl_binds/sdl_binds.h"

namespace tmd{
    /**
     * Tuple containing a detection.
     * A detection is characterized by the following attributes :
     *      _ A box around the detection.
     *      _ A set of boxes representing the body parts.
     *      _ A score.
     *      _ The index of the component from which the detection came from.
     * This typedef is just to make the code clearer.
     */
    typedef std::tuple<cv::Rect, std::vector<cv::Rect>, float, int> detection;

    /**
     *  DPM Class. Extract the players and their body parts from the given
     *  frame using the pedestrian model.
     *
     *  This is a custom version of the LatentSVMDetector from openCV.
     *  The reason is because in openCV, LatentSVMDetector cannot allow us to
     *  retrieve the body parts from the detections nor the component it is
     *  coming from.
     *  We thus made our own version of LatentSVMDetector allowing us to save
     *  the body parts and the origin component for each detected players.
     *  The code has not changed that much, however we had to take some
     *  function as is because our program could not find the declarations
     *  (the static functions in the .cpp files for example).
     */
    class DPM {
    public:
        /**
         * Constructor of the detector.
         * No arguments, every parameters are given in the configuration file.
         */
        DPM();

        /**
         * Destructor of the detector.
         */
        ~DPM();

        /**
         * Extract the players from the given frame. For each player detected
         * a player_t* structure is created. The following attributes are set
         * in every player_t* returned :
         *      _ frame_index.
         *      _ likelihood.
         *      _ mask_image.
         *      _ original_image.
         *      _ pos_frame.
         *      _ features : body_parts.
         *      _ features : torso.
         *      _ features : torso_mask.
         *      _ features : torso_pos.
         * All the player_t* are put into a vector, and this vector is then
         * returned.
         */
        std::vector<tmd::player_t*> extract_players_and_body_parts
                (tmd::frame_t* frame);

    private:
        /** The following functions are taken from the source code of the
         * LatentSVMDetector from openCV.
         */

        /*
        // find rectangular regions in the given image that are likely
        // to contain objects and corresponding confidence levels
        //
        // API
        // CvSeq* cvLatentSvmDetectObjects(const IplImage* image,
        //                                  CvLatentSvmDetector* detector,
        //                                  CvMemStorage* storage,
        //                                  float overlap_threshold = 0.5f,
                                            int numThreads = -1);
        // INPUT
        // image                - image to detect objects in
        // detector             - Latent SVM detector in internal representation
        // storage              - memory storage to store the resultant sequence
        //                          of the object candidate rectangles
        // overlap_threshold    - threshold for the non-maximum suppression
                                    algorithm.
        */
        void cvLatentSvmDetectObjects(IplImage* image,
                                        CvLatentSvmDetector* detector,
                                        CvMemStorage* storage,
                                        float overlap_threshold, int
                                        numThreads);


        /*
        // Computation root filters displacement and values of score function
        //
        // API
        // int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                       const filterObject **filters,
                                       int kComponents, const int *kPartFilters,
                                       const float *b, float scoreThreshold,
                                       CvPoint **points, CvPoint **oppPoints,
                                       float **score, int *kPoints);
        // INPUT
        // H                 - feature pyramid
        // filters           - filters (root filter then it's part filters)
        // kComponents       - root filters number
        // kPartFilters      - array of part filters number for each component
        // b                 - array of linear terms
        // scoreThreshold    - score threshold
        // OUTPUT
        // points            - root filters displacement (top left corners)
        // oppPoints         - root filters displacement (bottom right corners)
        // score             - array of score values
        // kPoints           - number of boxes
        // RESULT
        // Error status
        */
        int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H,
                                    const CvLSVMFilterObject **filters,
                                    int kComponents, const int *kPartFilters,
                                    const float *b, float scoreThreshold,
                                    CvPoint **points, CvPoint **oppPoints,
                                    float **score, int *kPoints,int numThreads);


        /*
        // Computation right bottom corners coordinates of bounding boxes
        //
        // API
        // int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                             int sizeX, int sizeY, CvPoint **oppositePoints);
        // INPUT
        // points            - left top corners coordinates of bounding boxes
        // levels            - levels of feature pyramid where points were found
        // (sizeX, sizeY)    - size of root filter
        // OUTPUT
        // oppositePoints    - right bottom corners coordinates of bounding
                                boxes
        // RESULT
        // Error status
        */
        int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                          int sizeX, int sizeY, CvPoint **oppositePoints);


        /*
        // Computation of the root filter displacement and values of score
            function
        //
        // API
        // int searchObjectThreshold(const featurePyramid *H,
                                 const filterObject **all_F, int n,
                                 float b,
                                 int maxXBorder, int maxYBorder,
                                 float scoreThreshold,
                                 CvPoint **points, int **levels, int *kPoints,
                                 float **score, CvPoint ***partsDisplacement);
        // INPUT
        // H                 - feature pyramid
        // all_F             - the set of filters (the first element is root
                               filter, other elements - part filters)
        // n                 - the number of part filters
        // b                 - linear term of the score function
        // maxXBorder        - the largest root filter size (X-direction)
        // maxYBorder        - the largest root filter size (Y-direction)
        // scoreThreshold    - score threshold
        // OUTPUT
        // points            - positions (x, y) of the upper-left corner
                               of root filter frame
        // levels            - levels that correspond to each position
        // kPoints           - number of positions
        // score             - values of the score function
        // partsDisplacement - part filters displacement for each position
                               of the root filter
        // RESULT
        // Error status
        */
        int searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                                  const CvLSVMFilterObject **all_F, int n,
                                  float b,
                                  int maxXBorder, int maxYBorder,
                                  float scoreThreshold,
                                  CvPoint **points, int **levels, int *kPoints,
                                  float **score, CvPoint ***partsDisplacement,
                                  int numThreads);


        /*
        // Transformation filter displacement from the block space
        // to the space of pixels at the initial image
        //
        // API
        // int convertPoints(int countLevel, int lambda,
                             int initialImageLevel,
                             CvPoint *points, int *levels,
                             CvPoint **partsDisplacement, int kPoints, int n,
                             int maxXBorder,
                             int maxYBorder);
        // INPUT
        // countLevel        - the number of levels in the feature pyramid
        // lambda            - method parameter
        // initialImageLevel - level of feature pyramid that contains feature
                                map for initial image
        // points            - the set of root filter positions
                                (in the block space)
        // levels            - the set of levels
        // partsDisplacement - displacement of part filters (in the block space)
        // kPoints           - number of root filter positions
        // n                 - number of part filters
        // maxXBorder        - the largest root filter size (X-direction)
        // maxYBorder        - the largest root filter size (Y-direction)
        // OUTPUT
        // points            - the set of root filter positions
                                    (in the space of pixels)
        // partsDisplacement - displacement of part filters
                                    (in the space of pixels)
        // RESULT
        // Error status
        */
        int convertPoints(int /*countLevel*/, int lambda,
                          int initialImageLevel,
                          CvPoint *points, int *levels,
                          CvPoint **partsDisplacement, int kPoints, int n,
                          int maxXBorder,
                          int maxYBorder);

        /*
        // Compute opposite point for filter box
        //
        // API
        // int getOppositePoint(CvPoint point,
                                int sizeX, int sizeY,
                                float step, int degree,
                                CvPoint *oppositePoint);

        // INPUT
        // point             - coordinates of filter top left corner
                               (in the space of pixels)
        // (sizeX, sizeY)    - filter dimension in the block space
        // step              - scaling factor
        // degree            - degree of the scaling factor
        // OUTPUT
        // oppositePoint     - coordinates of filter bottom corner
                               (in the space of pixels)
        // RESULT
        // Error status
        */
        int getOppositePoint(CvPoint point,
                             int sizeX, int sizeY,
                             float step, int degree,
                             CvPoint *oppositePoint);


        /*
        // Perform non-maximum suppression algorithm to remove "similar"
         bounding boxes
        //
        // API
        // int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold,
                             int *numBoxesout, CvPoint **pointsOut,
                             CvPoint **oppositePointsOut, float **scoreOut);
        // INPUT
        // numBoxes          - number of bounding boxes
        // points            - array of left top corner coordinates
        // oppositePoints    - array of right bottom corner coordinates
        // score             - array of detection scores
        // overlapThreshold  - threshold: bounding box is removed if overlap
                               part is greater than passed value
        // OUTPUT
        // numBoxesOut       - the number of bounding boxes algorithm returns
        // pointsOut         - array of left top corner coordinates
        // oppositePointsOut - array of right bottom corner coordinates
        // scoreOut          - array of detection scores
        // RESULT
        // Error status
        */
        int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                                  const CvPoint *oppositePoints,
                                  const float *score, float overlapThreshold,
                                  int *numBoxesOut, CvPoint **pointsOut,
                                  CvPoint **oppositePointsOut, float
                                  **scoreOut);

        /** End of openCV functions. */

        /**
         * Custom variation of the showPartFilterBoxes function (latentsvm.cpp).
         * This method returns the parts boxes in a vector of cv::Rects.
         *
         * filters : The filters of the detector.
         * n : The number of partDisplacement points.
         * partsDisplacement : The parts points.
         * level : Level of the partsDisplacement points.
         */
        std::vector<cv::Rect> get_parts_rect_for_point(const
                                            CvLSVMFilterObject **filters,
                                            int n, CvPoint
                                            *partsDisplacement, int
                                            level);

        /**
         * Clip the upper right corner of the boxes.
         *
         * width : Width of the image, ie maximum x coordinate.
         * height : Height of the image, ie maximum y coordinate.
         * points : The points.
         * kPoints : Number of points in the points pointer.
         */
        int clippingBoxesUpperRightCorner(int width, int height,
                                                   CvPoint *points, int
                                                   kPoints);

        /**
         * Clip the lower left corner of the boxes.
         *
         * width : Width of the image, ie maximum x coordinate.
         * height : Height of the image, ie maximum y coordinate.
         * points : The points.
         * kPoints : Number of points in the points pointer.
         */
        int clippingBoxesLowerLeftCorner(int width, int height,
                                                  CvPoint *points, int kPoints);

        /**
         * Clamp all the detections to the given width / height.
         */
        void clamp_detections(int width, int height);

        /**
         * Extract the torso of the given player_t* and set it directly.
         *
         * player : The player we want to extract the torso.
         * component_level : The component index (level) of the detection
         * corresponding to this player. For our model, the torso boxes are
         * not at the same indices according to the component level.
         */
        void extractTorsoForPlayer(player_t *player, int component_level);

        /**
         * The actual detector.
         */
        CvLatentSvmDetector *m_detector;

        /**
         * All the current detections.
         */
        std::vector<tmd::detection> m_detections;
    };
}

#endif //BACHELOR_PROJECT_FAST_DPM_H
