#include <opencv2/highgui/highgui.hpp>
#include "../headers/dpm_detector.h"
#include "../headers/debug.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"

namespace tmd {
    DPMDetector::DPMDetector(std::string model_file) {
        m_detector = cvLoadLatentSvmDetector(model_file.c_str());
        if (m_detector == NULL) {
            throw std::invalid_argument("Error in DPMDetector : couldn't create"
                                                " the detector.");
        }
        m_numthreads = TMD_DPM_DETECTOR_NUMTHREADS;
        int root_size_x = m_detector->filters[0]->sizeX;
        int root_size_y = m_detector->filters[0]->sizeY;
        tmd::debug("DPMDetector", "DPMDetector",
                   "Detector loaded, root filter size = (" +
                   std::to_string(root_size_x) + ", " +
                   std::to_string(root_size_y) + ").");
    }

    DPMDetector::~DPMDetector() {
        cvReleaseLatentSvmDetector(&m_detector); // Bug in CLion.
    }

    void DPMDetector::extractBodyParts(tmd::player_t *player) {
        if (player == NULL) {
            throw std::invalid_argument("Error in DPMDetector : NULL pointer in"
                                                " extractBodyParts method.");
        }
        // (If you're using CLion as an IDE, the following will be underline
        // in red as if it was not correct. However this is a bug from Clion,
        // you can safely ignore it, the line typechecks and compile just fine.
        IplImage playerImage = player->original_image;
        player->features.body_parts = getPartBoxesForImage(&playerImage);
        if (player->features.body_parts.size() > 0) {
            // Clip the boxes if needed.
            clipBoxes(player);
            // Shrink the player box to the minimum size.
            shrinkBox(player);
            // If there is a player on the image, we compute its torso box.
            extractTorsoForPlayer(player);
        }
    }

    /**
     * Redefinition of EstimateBoxes, because the code couldn't find it.
     * This is the exact same as the openCV implementation.
     */
    int DPMDetector::customEstimateBoxes(CvPoint *points, int *levels,
                                         int kPoints,
                                         int sizeX, int sizeY,
                                         CvPoint **oppositePoints) {
        int i;
        float step;

        step = powf(2.0f, 1.0f / ((float) (LAMBDA)));

        *oppositePoints = (CvPoint *) malloc(sizeof(CvPoint) * kPoints);
        for (i = 0; i < kPoints; i++) {
            getOppositePoint(points[i], sizeX, sizeY, step, levels[i] - LAMBDA,
                             &((*oppositePoints)[i]));
        }
        return LATENT_SVM_OK;
    }

    /*
    // Drawing part filter boxes
    //
    // API
    // int showPartFilterBoxes(const IplImage *image,
                               const filterObject *filter,
                               CvPoint *points, int *levels, int kPoints,
                               CvScalar color, int thickness,
                               int line_type, int shift);
    // INPUT
    // image             - initial image
    // filters           - a set of part filters
    // n                 - number of part filters
    // partsDisplacement - a set of points
    // levels            - levels of feature pyramid
    // kPoints           - number of foot filter positions
    // color             - line color for each box
    // thickness         - line thickness
    // line_type         - line type
    // shift             - shift
    // OUTPUT
    // parts             - reference on a vector ready to contain positions of
                                // all body parts.
    // window contained initial image and filter boxes
    // RESULT
    // Error status
    */
    float delta(float a, float b){
        return static_cast<float> (fabs(a -b));
    }
    int DPMDetector::detectBestPartBoxes(std::vector<cv::Rect> &parts,
                                         IplImage *image,
                                         const CvLSVMFilterObject **filters,
                                         int n,
                                         CvPoint **partsDisplacement,
                                         int *levels, int kPoints,
                                         float *scores) {
        int i, j;
        float step;
        CvPoint oppositePoint;

        step = powf(2.0f, 1.0f / ((float) LAMBDA));

        int max_level = 0;
        for (int i = 0; i < kPoints; i++) {
            if (levels[i] > max_level)
                max_level = levels[i];
        }
        float max_score = -999999;
        std::vector<float> score_vector;
        for (int i = 0; i < kPoints; i++) {
            score_vector.push_back(scores[i]);
            if (scores[i] > max_score)
                max_score = scores[i];
        }
        std::sort(score_vector.begin(), score_vector.end());
        float max_score_1 = score_vector[score_vector.size()-1];
        float max_score_2 = score_vector[score_vector.size()-2];
        float max_score_3 = score_vector[score_vector.size()-3];

        tmd::debug("DPMDetector", "detectBestPartBoxes", "KPoint (i max) = "
                                                     + std::to_string(kPoints));
        tmd::debug("DPMDetector", "detectBestPartBoxes", "n (j max) = "
                                                     + std::to_string(n));
        float epsilon = 0.01;
        for (i = 0; i < kPoints; i++) {
            for (j = 0; j < n; j++) {
                getOppositePoint(partsDisplacement[i][j],
                                 filters[j + 1]->sizeX, filters[j + 1]->sizeY,
                                 step, levels[i] - 2 * LAMBDA, &oppositePoint);

                if (scores[i] > -10) {
                    cv::Rect r(partsDisplacement[i][j], oppositePoint);
                    parts.push_back(r);
                }
            }
        }

        return LATENT_SVM_OK;
    }

    // Custom definition of cvLatentSvmDetectObjects (latentsvmDetector.cpp)
    std::vector<cv::Rect> DPMDetector::getPartBoxesForImage(IplImage *image) {
        CvLSVMFeaturePyramid *H = 0;
        CvPoint *points = 0, *oppPoints = 0;
        int kPoints = 0;
        float *score = 0;
        unsigned int maxXBorder = 0, maxYBorder = 0;
        int numBoxesOut = 0;
        CvPoint *pointsOut = 0;
        CvPoint *oppPointsOut = 0;
        float *scoreOut = 0;
        CvSeq *result_seq = 0;
        int error = 0;

        if (image->nChannels == 3)
            cvCvtColor(image, image, CV_BGR2RGB);

        // Getting maximum filter dimensions
        getMaxFilterDims((const CvLSVMFilterObject **) (m_detector->filters),
                         m_detector->num_components,
                         m_detector->num_part_filters, &maxXBorder,
                         &maxYBorder);
        // Create feature pyramid with nullable border
        tmd::debug("DPMDetector", "getPartBoxesForImage",
                   "create featurePyramid.");
        H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
        tmd::debug("DPMDetector", "getPartBoxesForImage", "done.");
        // Search object
        std::vector<cv::Rect> parts;
        error = preparePartDetection(parts, image, H,
                                     (const CvLSVMFilterObject **) (m_detector
                                             ->filters),
                                     m_detector->num_components,
                                     m_detector->num_part_filters,
                                     m_detector->b,
                                     m_detector->score_threshold,
                                     &points, &oppPoints,
                                     &score, &kPoints,
                                     m_numthreads);
        if (error != LATENT_SVM_OK) {
            parts.clear();
            return parts;
        }

        // removed from original source : (modified a bit (no CvSeq))
            float overlap_threshold =
                    0.0; //(added)
            // Clipping boxes
            clippingBoxes(image->width, image->height, points, kPoints);
            clippingBoxes(image->width, image->height, oppPoints, kPoints);
            // NMS procedure
            nonMaximumSuppression(kPoints, points, oppPoints, score, overlap_threshold,
                        &numBoxesOut, &pointsOut, &oppPointsOut, &scoreOut);

            for (int i = 0; i < numBoxesOut; i++)
            {
                CvRect bounding_box = {0, 0, 0, 0};
                bounding_box.x = pointsOut[i].x;
                bounding_box.y = pointsOut[i].y;
                bounding_box.width = oppPointsOut[i].x - pointsOut[i].x;
                bounding_box.height = oppPointsOut[i].y - pointsOut[i].y;
                cv::Rect rect = bounding_box;
                //parts.push_back(rect);
            }
         // End of removed block.

        if (image->nChannels == 3)
            cvCvtColor(image, image, CV_RGB2BGR);
        if (H == NULL)
            tmd::debug("DPMDetector", "getPartBoxesForImage", "pyramid is "
                    "null.");
        freeFeaturePyramidObject(&H);
        free(points);
        free(oppPoints);
        free(score);
        free(scoreOut);

        return parts;
    }


    /**
     * Redefinition of searchObjectThreshold from openCV.
     * This version is the exact same, but without the multi-threading support
     * which would return an error.
     */
    int customSearchObjectThreshold(const CvLSVMFeaturePyramid *H,
                                    const CvLSVMFilterObject **all_F, int n,
                                    float b,
                                    int maxXBorder, int maxYBorder,
                                    float scoreThreshold,
                                    CvPoint **points, int **levels,
                                    int *kPoints,
                                    float **score, CvPoint ***partsDisplacement,
                                    int numThreads) {
        int opResult;
        tmd::debug("DPMDetector", "customSearchObjectThreshold",
                   "call thresholdFunctionalScore()");
        opResult = thresholdFunctionalScore(all_F, n, H, b,
                                            maxXBorder, maxYBorder,
                                            scoreThreshold,
                                            score, points, levels,
                                            kPoints, partsDisplacement);
        tmd::debug("DPMDetector", "customSearchObjectThreshold", "done.");

        if (opResult != LATENT_SVM_OK) {
            return LATENT_SVM_SEARCH_OBJECT_FAILED;
        }

        // Transformation filter displacement from the block space
        // to the space of pixels at the initial image
        // that settles at the level number LAMBDA
        convertPoints(H->numLevels, LAMBDA, LAMBDA, (*points),
                      (*levels), (*partsDisplacement), (*kPoints), n,
                      maxXBorder, maxYBorder);

        return LATENT_SVM_OK;
    }


    // Custom definition of searchObjectThresholdSomeComponents (latentsvm.cpp)
    int DPMDetector::preparePartDetection(std::vector<cv::Rect> &parts,
                                          IplImage *image,
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
                                          int numThreads) {
        tmd::debug("DPMDetector", "preparePartDetection", "Entering method.");
        //int error = 0;
        int i, j, s, f, componentIndex;
        unsigned int maxXBorder, maxYBorder;
        CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
        float **scoreArr;
        int *kPointsArr, **levelsArr;

        // Allocation memory
        pointsArr = (CvPoint **) malloc(sizeof(CvPoint *) * kComponents);
        oppPointsArr = (CvPoint **) malloc(sizeof(CvPoint *) * kComponents);
        scoreArr = (float **) malloc(sizeof(float *) * kComponents);
        kPointsArr = (int *) malloc(sizeof(int) * kComponents);
        levelsArr = (int **) malloc(sizeof(int *) * kComponents);
        partsDisplacementArr = (CvPoint ***) malloc(
                sizeof(CvPoint **) * kComponents);

        // Getting maximum filter dimensions
        /*error = */getMaxFilterDims(filters, kComponents, kPartFilters,
                                     &maxXBorder, &maxYBorder);
        componentIndex = 0;
        *kPoints = 0;
        // For each component perform searching
        int i_max = kComponents - 1;
        for (i = 0; i < kComponents; i++) {
            tmd::debug("DPMDetector", "preparePartDetection",
                       "Call searchObjectThreshold");
                    // Same impl.
            int error = customSearchObjectThreshold(H,
                                                    &(filters[componentIndex]),
                                                    kPartFilters[i],
                                                    b[i], maxXBorder,
                                                    maxYBorder, scoreThreshold,
                                                    &(pointsArr[i]),
                                                    &(levelsArr[i]),
                                                    &(kPointsArr[i]),
                                                    &(scoreArr[i]),
                                                    &(partsDisplacementArr[i]),
                                                    numThreads);
            tmd::debug("DPMDetector", "preparePartDetection",
                       "searchObjectThreshold finished.");
            if (error != LATENT_SVM_OK) {
                // Release allocated memory
                free(pointsArr);
                free(oppPointsArr);
                free(scoreArr);
                free(kPointsArr);
                free(levelsArr);
                free(partsDisplacementArr);
                tmd::debug("DPMDetector", "preparePartDetection",
                           "searchObjectThreshold finished with error.");
                if (error == LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT) {
                    tmd::debug("DPMDetector", "preparePartDetection",
                           "error is LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT.");
                }
                return LATENT_SVM_SEARCH_OBJECT_FAILED;
            }
            // Same impl.
            customEstimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
                                filters[componentIndex]->sizeX,
                                filters[componentIndex]->sizeY,
                                &(oppPointsArr[i]));
            componentIndex += (kPartFilters[i] + 1);
            *kPoints += kPointsArr[i];
        }

        detectBestPartBoxes(parts, image, filters,
                            kPartFilters[i_max],
                            partsDisplacementArr[i_max],
                            levelsArr[i_max], kPointsArr[i_max],
                            scoreArr[i_max]);

        // Removed from original sources :
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
         // End of removed block.

        // Release allocated memory
        for (i = 0; i < kComponents; i++) {
            free(pointsArr[i]);
            free(oppPointsArr[i]);
            free(scoreArr[i]);
            free(levelsArr[i]);
            for (j = 0; j < kPointsArr[i]; j++) {
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
        tmd::debug("DPMDetector", "preparePartDetection",
                   "Exiting preparePartDetection method.");
        return LATENT_SVM_OK;
    }

    int max(int a, int b){
        if (a < b) return b;
        else return a;
    }

    void DPMDetector::extractTorsoForPlayer(player_t *player) {
        if (player == NULL){
            throw std::invalid_argument("Error null pointer given to "
                                            "extractTorsoForPlayer method");
        }
        else if (player->features.body_parts.size() < 3){
            throw std::invalid_argument("Error not enough body parts in "
                                                "extractTorsoForPlayer");
        }
        cv::Rect torso1 = player->features.body_parts[1];
        cv::Rect torso2 = player->features.body_parts[2];
        cv::Rect mean;
        mean.x = (torso1.x + torso2.x) / 2;
        mean.y = (torso1.y + torso2.y) / 2;
        int oppoX = ((torso1.x + torso1.width) + (torso2.x + torso2.width))/2;
        int oppoY = ((torso1.y + torso1.height) + (torso2.y + torso2.height))/2;
        mean.width = oppoX - mean.x;
        mean.height = oppoY - mean.y;
        std::cout << "Player box = " << player->pos_frame << std::endl;
        std::cout << "Mean box = " << mean << std::endl;
        std::cout << "Body parts 1 = " << torso1 << std::endl;
        std::cout << "Body parts 2 = " << torso2 << std::endl;
        assert(mean.x >= 0);
        assert(mean.y >= 0);
        assert(mean.y + mean.height <= player->pos_frame.height);
        assert(mean.x + mean.width <= player->pos_frame.width);
        cv::Rect roi = mean;
        cv::Rect m = player->pos_frame;
        assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.width &&
               0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.height);
        player->features.torso = (player->original_image.clone())(mean);
        player->features.torso_mask = (player->mask_image.clone())(mean);
        player->features.torso_pos = mean;
    }

    void DPMDetector::clipBoxes(player_t* player){
        size_t boxes_count = player->features.body_parts.size();

        for (size_t i = 0 ; i < boxes_count ; i ++){
            cv::Rect *part = &(player->features.body_parts[i]);

            // Zero boundaries
            if (part->x < 0) part->x = 0;
            if (part->y < 0) part->y = 0;

            // Right and bottom boundaries.
            if (part->x + part->width > player->pos_frame.width){
                part->width = player->pos_frame.width - part->x;
            }
            if (part->y + part->height > player->pos_frame.height){
                part->height = player->pos_frame.height - part->y;
            }
            assert(part->x >= 0);
            assert(part->y >= 0);
            /*assert(part->width >= 0);
            assert(part->height >= 0);*/
            assert(part->x + part->width <= player->pos_frame.width);
            assert(part->y + part->height <= player->pos_frame.height);
        }
    }

    void DPMDetector::shrinkBox(player_t* player){
        int min_x = std::numeric_limits<int>::max();
        int min_y = std::numeric_limits<int>::max();
        int max_x = 0;
        int max_y = 0;

        for (int i = 0 ; i < player->features.body_parts.size() ; i ++){
            cv::Rect part = player->features.body_parts[i];
            assert(part.x >= 0);
            assert(part.y >= 0);
            assert(part.x + part.width <= player->pos_frame.width);
            assert(part.y + part.height <= player->pos_frame.height);
            if (part.x < min_x) min_x = part.x;
            if (part.y < min_y) min_y = part.y;
            if (part.x + part.width > max_x) max_x = part.x + part.width;
            if (part.y + part.height > max_y) max_y = part.y + part.height;
        }
        std::cout << "DPMDetector::shrinkBox() : pos_frame = " <<
                player->pos_frame << std::endl;

        assert(min_x >= 0);
        assert(min_y >= 0);
        assert(max_x <= player->pos_frame.width);
        assert(max_y <= player->pos_frame.height);
        player->pos_frame.x += min_x;
        player->pos_frame.y += min_y;
        player->pos_frame.width = max_x - min_x;
        player->pos_frame.height = max_y - min_y;

        size_t boxes_count = player->features.body_parts.size();
        for (size_t i = 0 ; i < boxes_count ; i ++) {
            cv::Rect *part = &(player->features.body_parts[i]);
            part->x -= min_x;
            part->y -= min_y;
        }

            tmd::debug("DPMDetector", "shrinkBox", "min_x = " + std::to_string
                                                                    (min_x));
        tmd::debug("DPMDetector", "shrinkBox", "max_x = " + std::to_string
                (max_x));
        tmd::debug("DPMDetector", "shrinkBox", "min_y = " + std::to_string
                (min_y));
        tmd::debug("DPMDetector", "shrinkBox", "max_y = " + std::to_string
                (max_y));

        cv::Rect new_player_image_rect;
        new_player_image_rect.x = min_x;
        new_player_image_rect.y = min_y;
        new_player_image_rect.width = max_x - min_x;
        new_player_image_rect.height = max_y - min_y;

        cv::Mat new_player_image = player->original_image.clone()
                (new_player_image_rect);
        cv::Mat new_player_mask = player->mask_image.clone()
                (new_player_image_rect);

        player->original_image = new_player_image;
        player->mask_image = new_player_mask;
    }
}