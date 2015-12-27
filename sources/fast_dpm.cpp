#include "../headers/fast_dpm.h"
#include "../headers/frame_t.h"

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


    int FastDPM::showPartFilterBoxes(IplImage *image,
                            const CvLSVMFilterObject **filters,
                            int n, CvPoint **partsDisplacement,
                            int *levels, int kPoints,
                            CvScalar color, int thickness,
                            int line_type, int shift)
    {
        int i, j;
        float step;
        CvPoint oppositePoint;
        cv::Mat old = cv::Mat(image).clone();

        step = powf( 2.0f, 1.0f / ((float)LAMBDA));
        i = 0;
        for (i = kPoints - 1; i >= 0; i--){
            for (j = 0; j < n; j++)
            {
                // Drawing rectangles for part filters
                getOppositePoint(partsDisplacement[i][j],
                                 filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                                 step, levels[i] - 2 * LAMBDA, &oppositePoint);
                cv::rectangle(old, partsDisplacement[i][j], oppositePoint,
                            color, thickness, line_type, shift);
            }
            cv::imshow("Initial image", old);
            std::cout << i << std::endl;
            cv::waitKey(0);
            old = cv::Mat(image).clone();
        }

        return LATENT_SVM_OK;
    }


    std::vector<cv::Rect> get_parts_rect_for_point(const CvLSVMFilterObject **filters,
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

    std::vector<std::vector<cv::Rect>> detectBestPartBoxes(
                                         const CvLSVMFilterObject **filters,
                                         int n,
                                         CvPoint **partsDisplacement,
                                         int *levels, int kPoints,
                                         float *scores) {
        int i, j;
        float step;
        CvPoint oppositePoint;

        step = powf(2.0f, 1.0f / ((float) LAMBDA));
        std::vector<std::vector<cv::Rect>> all_parts;
        for (i = 0; i < kPoints; i++) {
            std::vector<cv::Rect> tmp_parts;
            for (j = 0; j < n; j++) {
                getOppositePoint(partsDisplacement[i][j],
                                 filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                                 step, levels[i] - 2 * LAMBDA, &oppositePoint);

                if (scores[i] > tmd::Config::dpm_extractor_score_threshold) {
                    cv::Rect r(partsDisplacement[i][j], oppositePoint);
                    tmp_parts.push_back(r);
                }
            }
            all_parts.push_back(tmp_parts);
        }

        return all_parts;
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
        int i_max = kComponents - 1;

        m_parts = detectBestPartBoxes(filters, kPartFilters[i_max],
                                      partsDisplacementArr[i_max],
                                      levelsArr[i_max], kPointsArr[i_max],
                                      scoreArr[i_max]);

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
            }
            s = f;
        }

        for (std::vector<cv::Rect> parts : m_parts){
            for (cv::Rect part : parts){
                cv::rectangle(m_image, part, m_color, 1, m_line_type, m_shift);
            }
            cv::imshow("Frame", m_image);
            cv::waitKey(0);
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