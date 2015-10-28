#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "../headers/dpm_detector.h"
#include "../headers/debug.h"

namespace tmd{
    DPMDetector::DPMDetector(std::string model_file){
        m_detector = cvLoadLatentSvmDetector(model_file.c_str());
        if (m_detector == NULL){
            throw std::invalid_argument("Error in DPMDetector : couldn't create"
                                                " the detector.");
        }
    }

    DPMDetector::~DPMDetector() {
        cvReleaseLatentSvmDetector(&m_detector);
    }

    void extractTorso(tmd::player_t* player){
        if (player == NULL){
            throw std::invalid_argument("Error in DPMDetector : NULL pointer in"
                                                " extractTorso method.");
        }

        // TODO : extract torso and update features of the player.
    }

    static int DPMDetector::CustomestimateBoxes(CvPoint *points, int *levels, int kPoints,
                             int sizeX, int sizeY, CvPoint **oppositePoints)
    {
        int i;
        float step;

        step = powf( 2.0f, 1.0f / ((float)(LAMBDA)));

        *oppositePoints = (CvPoint *)malloc(sizeof(CvPoint) * kPoints);
        for (i = 0; i < kPoints; i++)
        {
            getOppositePoint(points[i], sizeX, sizeY, step, levels[i] - LAMBDA, &((*oppositePoints)[i]));
        }
        return LATENT_SVM_OK;
    }

    int DPMDetector::CustomshowPartFilterBoxes(IplImage *image,
                                  const CvLSVMFilterObject **filters,
                                  int n, CvPoint **partsDisplacement,
                                  int *levels, int kPoints,
                                  CvScalar color, int thickness,
                                  int line_type, int shift, float* scores){
        int i, j;
        float step;
        CvPoint oppositePoint;

        step = powf( 2.0f, 1.0f / ((float)LAMBDA));

        float th = 30;

        int max_level = 0;
        for (int i = 0 ; i < kPoints ; i ++){if (levels[i] > max_level) max_level = levels[i];}
        float max_score_for_level = -2.f;
        for (int i = 0 ; i < kPoints ; i ++){if (levels[i] == max_level && scores[i] > max_score_for_level) max_score_for_level = scores[i];}
        tmd::debug("max_level = " + std::to_string(max_level));
        tmd::debug("max_score_for_level = " + std::to_string(max_score_for_level));
        for (int i = 0 ; i < kPoints ; i ++){if (levels[i] == max_level) tmd::debug("    Max level ; score = "  +std::to_string(scores[i]));}
        for (i = 0; i < kPoints; i++)
        {
            for (j = 0; j < n; j++)
            {
                // Drawing rectangles for part filters
                getOppositePoint(partsDisplacement[i][j],
                                 filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                                 step, levels[i] - 2 * LAMBDA, &oppositePoint);
                float dist = pow(partsDisplacement[i][j].x - oppositePoint.x, 2) + pow(partsDisplacement[i][j].y - oppositePoint.y, 2);
                tmd::debug("score = " + std::to_string(scores[i]));
                if (levels[i]  == max_level  && scores[i] == max_score_for_level) {
                    tmd::debug("DRAAAAW");
                    cvRectangle(image, partsDisplacement[i][j], oppositePoint,
                                color, thickness, line_type, shift);
                    cvShowImage("Initial image", image);
                    cv::waitKey(0);
                }
            }
        }

        return LATENT_SVM_OK;
    }

    int DPMDetector::CustomsearchObjectThresholdSomeComponents(IplImage* image, const CvLSVMFeaturePyramid *H,
                                                  const CvLSVMFilterObject **filters,
                                                  int kComponents, const int *kPartFilters,
                                                  const float *b, float scoreThreshold,
                                                  CvPoint **points, CvPoint **oppPoints,
                                                  float **score, int *kPoints,
                                                  int numThreads){
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
        float** scores = new float*[kComponents];
        int i_max = kComponents - 1;
        for (i = 0; i < kComponents; i++)
        {
            int error = searchObjectThreshold(H, &(filters[componentIndex]), kPartFilters[i],
                                              b[i], maxXBorder, maxYBorder, scoreThreshold,
                                              &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
                                              &(scoreArr[i]), &(partsDisplacementArr[i]), numThreads);
            /*CvScalar color;
            color.val[0] = 255; color.val[1] = 255; color.val[2] = 0; color.val[3] = 255;
            const int thickness = 1;
            const int line_type = 8; // 8 connected line.
            const int shift = 0;
            bool draw = true;

            tmd::debug("New draw");
            if (draw) CustomshowPartFilterBoxes(image, filters,
                                                kPartFilters[i], partsDisplacementArr[i],
                                    levelsArr[i], kPointsArr[i],
                                    color, thickness,
                                    line_type, shift, scoreArr[i]);*/



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
            CustomestimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
                          filters[componentIndex]->sizeX, filters[componentIndex]->sizeY, &(oppPointsArr[i]));
            componentIndex += (kPartFilters[i] + 1);
            *kPoints += kPointsArr[i];
        }

        CvScalar color;
        color.val[0] = 255; color.val[1] = 255; color.val[2] = 0; color.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        bool draw = true;

        tmd::debug("New draw");
        if (draw) CustomshowPartFilterBoxes(image, filters,
                                            kPartFilters[i_max], partsDisplacementArr[i_max],
                                            levelsArr[i_max], kPointsArr[i_max],
                                            color, thickness,
                                            line_type, shift, scoreArr[i_max]);

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

    CvSeq* DPMDetector::CustomcvLatentSvmDetectObjects(IplImage* image,
                                          CvLatentSvmDetector* detector,
                                          CvMemStorage* storage,
                                          float overlap_threshold, int numThreads){
        CvPoint*** partsDisplacementArr = NULL;
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
        error = CustomsearchObjectThresholdSomeComponents(image, H, (const CvLSVMFilterObject**)(detector->filters),
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
}