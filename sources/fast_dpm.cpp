#include "../headers/fast_dpm.h"
#include "../headers/frame_t.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

namespace tmd{

    FastDPM::FastDPM(){
        m_detector = cvLoadLatentSvmDetector(tmd::Config::model_file_path.c_str());
        m_color.val[0] = 255;
        m_color.val[0] = 255;
        m_color.val[0] = 0;
        m_color.val[0] = 255;
        m_line_type = 8;
        m_shift = 0;
    }

    std::vector<tmd::player_t*> FastDPM::extract_players_and_body_parts(
            cv::Mat blob){
        m_image = blob;
        IplImage blobImage = m_image;
        CvMemStorage *memStorage = cvCreateMemStorage(0);

        this->cvLatentSvmDetectObjects(&blobImage, m_detector, memStorage,
                                       tmd::Config::dpm_extractor_overlapping_threshold, 1);
    }


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
    // overlap_threshold    - threshold for the non-maximum suppression algorithm [here will be the reference to original paper]
    // OUTPUT
    // sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
    */
    CvSeq* FastDPM::cvLatentSvmDetectObjects(IplImage* image,
                                    CvLatentSvmDetector* detector,
                                    CvMemStorage* storage,
                                    float overlap_threshold, int numThreads)
    {
        CvLSVMFeaturePyramid *H = 0;
        CvPoint *points = 0, *oppPoints = 0;
        int kPoints = 0;
        float *score = 0;
        unsigned int maxXBorder = 0, maxYBorder = 0;
        int numBoxesOut = 0;
        CvPoint *pointsOut = 0;
        CvPoint *oppPointsOut = 0;
        float *scoreOut = 0;
        CvSeq* result_seq = 0;
        int error = 0;

        if(image->nChannels == 3)
            cvCvtColor(image, image, CV_BGR2RGB);

        // Getting maximum filter dimensions
        getMaxFilterDims((const CvLSVMFilterObject**)(detector->filters), detector->num_components,
                         detector->num_part_filters, &maxXBorder, &maxYBorder);
        // Create feature pyramid with nullable border
        H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
        // Search object
        error = this->searchObjectThresholdSomeComponents(H, (const
                                                     CvLSVMFilterObject**)(detector->filters),
                                                    detector->num_components, detector->num_part_filters, detector->b, detector->score_threshold,
                                                    &points, &oppPoints, &score, &kPoints, numThreads);
        if (error != LATENT_SVM_OK)
        {
            return NULL;
        }
        // Clipping boxes
        clippingBoxes(image->width, image->height, points, kPoints);
        clippingBoxes(image->width, image->height, oppPoints, kPoints);
        // NMS procedure
        nonMaximumSuppression(kPoints, points, oppPoints, score, overlap_threshold,
                              &numBoxesOut, &pointsOut, &oppPointsOut, &scoreOut);

        for (auto t : m_detections){
            cv::Mat clone = m_image.clone();
            for (cv::Rect part : std::get<2>(t)){
                cv::rectangle(clone, part, m_color, 1, m_line_type, m_shift);
            }
            cv::imshow("Frame", clone);
            cv::waitKey(0);
        }

        result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvObjectDetection), storage );

        for (int i = 0; i < numBoxesOut; i++)
        {
            CvObjectDetection detection = {{0, 0, 0, 0}, 0};
            detection.score = scoreOut[i];
            CvRect bounding_box = {0, 0, 0, 0};
            bounding_box.x = pointsOut[i].x;
            bounding_box.y = pointsOut[i].y;
            bounding_box.width = oppPointsOut[i].x - pointsOut[i].x;
            bounding_box.height = oppPointsOut[i].y - pointsOut[i].y;
            detection.rect = bounding_box;
            cvSeqPush(result_seq, &detection);
        }

        if(image->nChannels == 3)
            cvCvtColor(image, image, CV_RGB2BGR);

        freeFeaturePyramidObject(&H);
        free(points);
        free(oppPoints);
        free(score);
        free(scoreOut);

        return result_seq;
    }

    /*
// Elimination boxes that are outside the image boudaries
//
// API
// int clippingBoxes(int width, int height,
                     CvPoint *points, int kPoints);
// INPUT
// width             - image wediht
// height            - image heigth
// points            - a set of points (coordinates of top left or
                       bottom right corners)
// kPoints           - points number
// OUTPUT
// points            - updated points (if coordinates less than zero then
                       set zero coordinate, if coordinates more than image
                       size then set coordinates equal image size)
// RESULT
// Error status
*/
    int FastDPM::clippingBoxes(int width, int height,
                      CvPoint *points, int kPoints)
    {
        int i;
        for (i = 0; i < kPoints; i++)
        {
            assert(points[i].x == m_points[i].x);
            assert(points[i].y == m_points[i].y);
            if (points[i].x > width - 1)
            {
                points[i].x = width - 1;
            }
            if (points[i].x < 0)
            {
                points[i].x = 0;
            }
            if (points[i].y > height - 1)
            {
                points[i].y = height - 1;
            }
            if (points[i].y < 0)
            {
                points[i].y = 0;
            }
        }
        return LATENT_SVM_OK;
    }


    static void sort(int n, const float* x, int* indices)
    {
        int i, j;
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++)
            {
                if (x[indices[j]] > x[indices[i]])
                {
                    //float x_tmp = x[i];
                    int index_tmp = indices[i];
                    //x[i] = x[j];
                    indices[i] = indices[j];
                    //x[j] = x_tmp;
                    indices[j] = index_tmp;
                }
            }
    }


    /*
// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
//
// API
// int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold,
                             int *numBoxesOut, CvPoint **pointsOut,
                             CvPoint **oppositePointsOut, float **scoreOut);
// INPUT
// numBoxes          - number of bounding boxes
// points            - array of left top corner coordinates
// oppositePoints    - array of right bottom corner coordinates
// score             - array of detection scores
// overlapThreshold  - threshold: bounding box is removed if overlap part
                       is greater than passed value
// OUTPUT
// numBoxesOut       - the number of bounding boxes algorithm returns
// pointsOut         - array of left top corner coordinates
// oppositePointsOut - array of right bottom corner coordinates
// scoreOut          - array of detection scores
// RESULT
// Error status
*/
    int FastDPM::nonMaximumSuppression(int numBoxes, const CvPoint *points,
                              const CvPoint *oppositePoints, const float *score,
                              float overlapThreshold,
                              int *numBoxesOut, CvPoint **pointsOut,
                              CvPoint **oppositePointsOut, float **scoreOut)
    {
        int i, j, index;
        float* box_area = (float*)malloc(numBoxes * sizeof(float));
        int* indices = (int*)malloc(numBoxes * sizeof(int));
        int* is_suppressed = (int*)malloc(numBoxes * sizeof(int));

        for (i = 0; i < numBoxes; i++)
        {
            indices[i] = i;
            is_suppressed[i] = 0;
            box_area[i] = (float)( (oppositePoints[i].x - points[i].x + 1) *
                                   (oppositePoints[i].y - points[i].y + 1));
        }

        sort(numBoxes, score, indices);
        for (i = 0; i < numBoxes; i++)
        {
            std::cout << points[indices[i]].x << " " << points[indices[i]].y
            << " vs " << m_points[indices[i]].x << " " <<
                    m_points[indices[i]].y << std::endl;
            if (!is_suppressed[indices[i]])
            {
                for (j = i + 1; j < numBoxes; j++)
                {
                    if (!is_suppressed[indices[j]])
                    {
                        int x1max = max(points[indices[i]].x, points[indices[j]].x);
                        int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);
                        int y1max = max(points[indices[i]].y, points[indices[j]].y);
                        int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);
                        int overlapWidth = x2min - x1max + 1;
                        int overlapHeight = y2min - y1max + 1;
                        if (overlapWidth > 0 && overlapHeight > 0)
                        {
                            float overlapPart = (overlapWidth * overlapHeight) / box_area[indices[j]];
                            if (overlapPart > overlapThreshold)
                            {
                                is_suppressed[indices[j]] = 1;
                                std::cout << "Erase point : " <<
                                        points[indices[j]].x << " " <<
                                        points[indices[j]].y << std::endl;
                                assert(m_scores[indices[j]] ==
                                               score[indices[j]]);
                                assert(m_points[indices[j]].x ==
                                               points[indices[j]].x);
                                 assert(m_points[indices[j]].y ==
                                               points[indices[j]].y);
                                m_parts.erase(m_parts.begin() + j);
                            }
                        }
                    }
                }
            }
        }

        *numBoxesOut = 0;
        for (i = 0; i < numBoxes; i++)
        {
            if (!is_suppressed[i]) (*numBoxesOut)++;
        }

        *pointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
        *oppositePointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
        *scoreOut = (float *)malloc((*numBoxesOut) * sizeof(float));
        index = 0;
        for (i = 0; i < numBoxes; i++)
        {
            if (!is_suppressed[indices[i]])
            {
                (*pointsOut)[index].x = points[indices[i]].x;
                (*pointsOut)[index].y = points[indices[i]].y;
                (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
                (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
                (*scoreOut)[index] = score[indices[i]];
                index++;
            }

        }

        free(indices);
        free(box_area);
        free(is_suppressed);

        return LATENT_SVM_OK;
    }


    std::vector<cv::Rect> FastDPM::get_parts_rect_for_point(const
                                                        CvLSVMFilterObject **filters,
                                                   int n, CvPoint
                                                   *partsDisplacement, int
                                                   level){
        int j;
        float step;
        CvPoint oppositePoint;
        std::vector<cv::Rect> parts;

        step = powf( 2.0f, 1.0f / ((float)LAMBDA));
        for (j = 0; j < n; j++) {
            // Drawing rectangles for part filters
            getOppositePoint(partsDisplacement[j],
                             filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                             step, level - 2 * LAMBDA, &oppositePoint);
            cv::Rect rect(partsDisplacement[j], oppositePoint);
            parts.push_back(rect);
        }
        return parts;
    }


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
    // filters           - filters (root filter then it's part filters, etc.)
    // kComponents       - root filters number
    // kPartFilters      - array of part filters number for each component
    // b                 - array of linear terms
    // scoreThreshold    - score threshold
    // OUTPUT
    // points            - root filters displacement (top left corners)  Les
     boites
    // oppPoints         - root filters displacement (bottom right corners)
     Les boites
    // score             - array of score values
    // kPoints           - number of boxes
    // RESULT
    // Error status
    */
    int FastDPM::searchObjectThresholdSomeComponents(const
                                                    CvLSVMFeaturePyramid *H,
                                            const CvLSVMFilterObject **filters,
                                            int kComponents, const int *kPartFilters,
                                            const float *b, float scoreThreshold,
                                            CvPoint **points, CvPoint **oppPoints,
                                            float **score, int *kPoints,
                                            int numThreads)
    {
        //int error = 0;
        int i, j, s, f, componentIndex;
        unsigned int maxXBorder, maxYBorder;
        CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
        float **scoreArr;
        int *kPointsArr, **levelsArr;

        // Allocation memory
        pointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
        oppPointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
        scoreArr = (float **)malloc(sizeof(float *) * kComponents);
        kPointsArr = (int *)malloc(sizeof(int) * kComponents);
        levelsArr = (int **)malloc(sizeof(int *) * kComponents);
        partsDisplacementArr = (CvPoint ***)malloc(sizeof(CvPoint **) * kComponents);

        // Getting maximum filter dimensions
        /*error = */getMaxFilterDims(filters, kComponents, kPartFilters, &maxXBorder, &maxYBorder);
        componentIndex = 0;
        *kPoints = 0;
        // For each component perform searching
        for (i = 0; i < kComponents; i++)
        {
            int error = this->searchObjectThreshold(H, &
                                                           (filters[componentIndex]), kPartFilters[i],
                                              b[i], maxXBorder, maxYBorder, scoreThreshold,
                                              &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
                                              &(scoreArr[i]), &(partsDisplacementArr[i]), numThreads);
            if (error != LATENT_SVM_OK)
            {
                // Release allocated memory
                free(pointsArr);
                free(oppPointsArr);
                free(scoreArr);
                free(kPointsArr);
                free(levelsArr);
                free(partsDisplacementArr);
                return LATENT_SVM_SEARCH_OBJECT_FAILED;
            }
            this->estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
                          filters[componentIndex]->sizeX, filters[componentIndex]->sizeY, &(oppPointsArr[i]));
            componentIndex += (kPartFilters[i] + 1);
            *kPoints += kPointsArr[i];
            /*IplImage image = m_image;

            this->showPartFilterBoxes(&image, filters, kPartFilters[i],
                                      partsDisplacementArr[i], levelsArr[i],
                                      kPointsArr[i], m_color, 1, m_line_type,
                                      m_shift);*/
        }

        *points = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
        *oppPoints = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
        *score = (float *)malloc(sizeof(float) * (*kPoints));
        s = 0;
        for (i = 0; i < kComponents; i++)
        {
            f = s + kPointsArr[i];
            for (j = s; j < f; j++)
            {
                (*points)[j].x = pointsArr[i][j - s].x;
                (*points)[j].y = pointsArr[i][j - s].y;
                (*oppPoints)[j].x = oppPointsArr[i][j - s].x;
                (*oppPoints)[j].y = oppPointsArr[i][j - s].y;
                (*score)[j] = scoreArr[i][j - s];
                std::vector<cv::Rect> p = get_parts_rect_for_point
                        (filters, kPartFilters[i],
                         partsDisplacementArr[i][j - s], levelsArr[i][j-s]);
                m_parts.push_back(p);
                m_scores.push_back((*score)[j]);
                m_points.push_back((*points)[j]);
            }
            s = f;
        }

        // Release allocated memory
        for (i = 0; i < kComponents; i++)
        {
            free(pointsArr[i]);
            free(oppPointsArr[i]);
            free(scoreArr[i]);
            free(levelsArr[i]);
            for (j = 0; j < kPointsArr[i]; j++)
            {
                free(partsDisplacementArr[i][j]);
            }
            free(partsDisplacementArr[i]);
        }
        free(pointsArr);
        free(oppPointsArr);
        free(scoreArr);
        free(kPointsArr);
        free(levelsArr);
        free(partsDisplacementArr);
        return LATENT_SVM_OK;
    }


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
    // oppositePoints     - right bottom corners coordinates of bounding boxes
    // RESULT
    // Error status
    */
    int FastDPM::estimateBoxes(CvPoint *points, int *levels, int kPoints,
                      int sizeX, int sizeY, CvPoint **oppositePoints)
    {
        int i;
        float step;

        step = powf( 2.0f, 1.0f / ((float)(LAMBDA)));

        *oppositePoints = (CvPoint *)malloc(sizeof(CvPoint) * kPoints);
        for (i = 0; i < kPoints; i++)
        {
            this->getOppositePoint(points[i], sizeX, sizeY, step, levels[i] -
                                                             LAMBDA, &((*oppositePoints)[i]));
        }
        return LATENT_SVM_OK;
    }

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
    int FastDPM::getOppositePoint(CvPoint point,
                         int sizeX, int sizeY,
                         float step, int degree,
                         CvPoint *oppositePoint)
    {
        float scale;
        scale = SIDE_LENGTH * powf(step, (float)degree);
        oppositePoint->x = (int)(point.x + sizeX * scale);
        oppositePoint->y = (int)(point.y + sizeY * scale);
        return LATENT_SVM_OK;
    }



    /*
    // Computation of the root filter displacement and values of score function
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
    // all_F             - the set of filters (the first element is root filter,
                           other elements - part filters)
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
    int FastDPM::searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                              const CvLSVMFilterObject **all_F, int n,
                              float b,
                              int maxXBorder, int maxYBorder,
                              float scoreThreshold,
                              CvPoint **points, int **levels, int *kPoints,
                              float **score, CvPoint ***partsDisplacement,
                              int numThreads)
    {
        int opResult;


        // Matching
#ifdef HAVE_TBB
        if (numThreads <= 0)
    {
        opResult = LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT;
        return opResult;
    }
    opResult = tbbThresholdFunctionalScore(all_F, n, H, b, maxXBorder, maxYBorder,
                                           scoreThreshold, numThreads, score,
                                           points, levels, kPoints,
                                           partsDisplacement);
#else
        opResult = this->thresholdFunctionalScore(all_F, n, H, b,
                                            maxXBorder, maxYBorder,
                                            scoreThreshold,
                                            score, points, levels,
                                            kPoints, partsDisplacement);

        (void)numThreads;
#endif
        if (opResult != LATENT_SVM_OK)
        {
            return LATENT_SVM_SEARCH_OBJECT_FAILED;
        }

        // Transformation filter displacement from the block space
        // to the space of pixels at the initial image
        // that settles at the level number LAMBDA
        this->convertPoints(H->numLevels, LAMBDA, LAMBDA, (*points),
                      (*levels), (*partsDisplacement), (*kPoints), n,
                      maxXBorder, maxYBorder);

        return LATENT_SVM_OK;
    }


    /*
    // Transformation filter displacement from the block space
    // to the space of pixels at the initial image
    //
    // API
    // int convertPoints(int countLevel, CvPoint *points, int *levels,
                      CvPoint **partsDisplacement, int kPoints, int n);
    // INPUT
    // countLevel        - the number of levels in the feature pyramid
    // points            - the set of root filter positions (in the block space)
    // levels            - the set of levels
    // partsDisplacement - displacement of part filters (in the block space)
    // kPoints           - number of root filter positions
    // n                 - number of part filters
    // initialImageLevel - level that contains features for initial image
    // maxXBorder        - the largest root filter size (X-direction)
    // maxYBorder        - the largest root filter size (Y-direction)
    // OUTPUT
    // points            - the set of root filter positions (in the space of pixels)
    // partsDisplacement - displacement of part filters (in the space of pixels)
    // RESULT
    // Error status
    */
    int FastDPM::convertPoints(int /*countLevel*/, int lambda,
                      int initialImageLevel,
                      CvPoint *points, int *levels,
                      CvPoint **partsDisplacement, int kPoints, int n,
                      int maxXBorder,
                      int maxYBorder)
    {
        int i, j, bx, by;
        float step, scale;
        step = powf( 2.0f, 1.0f / ((float)lambda) );

        computeBorderSize(maxXBorder, maxYBorder, &bx, &by);

        for (i = 0; i < kPoints; i++)
        {
            // scaling factor for root filter
            scale = SIDE_LENGTH * powf(step, (float)(levels[i] - initialImageLevel));
            points[i].x = (int)((points[i].x - bx + 1) * scale);
            points[i].y = (int)((points[i].y - by + 1) * scale);

            // scaling factor for part filters
            scale = SIDE_LENGTH * powf(step, (float)(levels[i] - lambda - initialImageLevel));
            for (j = 0; j < n; j++)
            {
                partsDisplacement[i][j].x = (int)((partsDisplacement[i][j].x -
                                                   2 * bx + 1) * scale);
                partsDisplacement[i][j].y = (int)((partsDisplacement[i][j].y -
                                                   2 * by + 1) * scale);
            }
        }
        return LATENT_SVM_OK;
    }


    /*
    // Computation score function that exceed threshold
    //
    // API
    // int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                                    const featurePyramid *H,
                                    float b,
                                    int maxXBorder, int maxYBorder,
                                    float scoreThreshold,
                                    float **score,
                                    CvPoint **points, int **levels, int *kPoints,
                                    CvPoint ***partsDisplacement);
    // INPUT
    // all_F             - the set of filters (the first element is root filter,
                           the other - part filters)
    // n                 - the number of part filters
    // H                 - feature pyramid
    // b                 - linear term of the score function
    // maxXBorder        - the largest root filter size (X-direction)
    // maxYBorder        - the largest root filter size (Y-direction)
    // scoreThreshold    - score threshold
    // OUTPUT
    // score             - score function values that exceed threshold
    // points            - the set of root filter positions (in the block space)
    // levels            - the set of levels
    // kPoints           - number of root filter positions
    // partsDisplacement - displacement of part filters (in the block space)
    // RESULT
    // Error status
    */
    int FastDPM::thresholdFunctionalScore(const CvLSVMFilterObject **all_F,
                                         int n,
                                 const CvLSVMFeaturePyramid *H,
                                 float b,
                                 int maxXBorder, int maxYBorder,
                                 float scoreThreshold,
                                 float **score,
                                 CvPoint **points, int **levels, int *kPoints,
                                 CvPoint ***partsDisplacement)
    {
        int l, i, j, k, s, f, level, numLevels;
        float **tmpScore;
        CvPoint ***tmpPoints;
        CvPoint ****tmpPartsDisplacement;
        int *tmpKPoints;
        int res;

        /* DEBUG
        FILE *file;
        //*/

        // Computation the number of levels for seaching object,
        // first lambda-levels are used for computation values
        // of score function for each position of root filter
        numLevels = H->numLevels - LAMBDA;

        // Allocation memory for values of score function for each level
        // that exceed threshold
        tmpScore = (float **)malloc(sizeof(float*) * numLevels);
        // Allocation memory for the set of points that corresponds
        // to the maximum of score function
        tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
        for (i = 0; i < numLevels; i++)
        {
            tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
        }
        // Allocation memory for memory for saving parts displacement on each level
        tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
        for (i = 0; i < numLevels; i++)
        {
            tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
        }
        // Number of points that corresponds to the maximum
        // of score function on each level
        tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
        for (i = 0; i < numLevels; i++)
        {
            tmpKPoints[i] = 0;
        }

        // Computation maxima of score function on each level
        // and getting the maximum on all levels
        /* DEBUG: maxScore
        file = fopen("maxScore.csv", "w+");
        fprintf(file, "%i;%lf;\n", H->lambda, tmpScore[0]);
        //*/
        (*kPoints) = 0;
        for (l = LAMBDA; l < H->numLevels; l++)
        {
            k = l - LAMBDA;
            //printf("Score at the level %i\n", l);
            res = thresholdFunctionalScoreFixedLevel(all_F, n, H, l, b,
                                                     maxXBorder, maxYBorder, scoreThreshold,
                                                     &(tmpScore[k]),
                                                     tmpPoints[k],
                                                     &(tmpKPoints[k]),
                                                     tmpPartsDisplacement[k]);
            //fprintf(file, "%i;%lf;\n", l, tmpScore[k]);
            if (res != LATENT_SVM_OK)
            {
                continue;
            }
            (*kPoints) += tmpKPoints[k];
        }
        //fclose(file);

        // Allocation memory for levels
        (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
        // Allocation memory for the set of points
        (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
        // Allocation memory for parts displacement
        (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
        // Allocation memory for score function values
        (*score) = (float *)malloc(sizeof(float) * (*kPoints));

        // Filling the set of points, levels and parts displacement
        s = 0;
        f = 0;
        for (i = 0; i < numLevels; i++)
        {
            // Computation the number of level
            level = i + LAMBDA;

            // Addition a set of points
            f += tmpKPoints[i];
            for (j = s; j < f; j++)
            {
                (*levels)[j] = level;
                (*points)[j] = (*tmpPoints[i])[j - s];
                (*score)[j] = tmpScore[i][j - s];
                (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
            }
            s = f;
        }

        // Release allocated memory
        for (i = 0; i < numLevels; i++)
        {
            free(tmpPoints[i]);
            free(tmpPartsDisplacement[i]);
        }
        free(tmpPoints);
        free(tmpScore);
        free(tmpKPoints);
        free(tmpPartsDisplacement);

        return LATENT_SVM_OK;
    }





    /*void FastDPM::display(CvPoint* points, int kPoints, CvPoint
    **partsDisplacement, int n){
        for (int i = 0 ; i < kPoints ; i ++){
            cv::Rect rect;
            for (int p = 0 ; p < n ; p ++){
                rect.x = points[i].x + partsDisplacement[i][p].x;
                rect.y = points[i].y + partsDisplacement[i][p].y;

            }
        }
    }*/
}