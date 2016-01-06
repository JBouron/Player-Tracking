#include "../../headers/features_extraction/dpm.h"
#include "../../headers/data_structures/frame_t.h"

#ifndef max
#define max(a, b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b)            (((a) < (b)) ? (a) : (b))
#endif

namespace tmd {

    DPM::DPM() {
        m_detector = cvLoadLatentSvmDetector(Config::model_file_path.c_str());
    }

    DPM::~DPM() {
        cvReleaseLatentSvmDetector(&m_detector);
    }

    std::vector<tmd::player_t *> DPM::extract_players_and_body_parts(
            tmd::frame_t *frame) {
        IplImage blobImage;
        if (tmd::Config::use_colored_mask_in_dpm){
            blobImage = frame->colored_mask_frame;
        }
        else{
            blobImage = frame->original_frame;
        }
        CvMemStorage *memStorage = cvCreateMemStorage(0);

        this->cvLatentSvmDetectObjects(&blobImage, m_detector, memStorage,
tmd::Config::dpm_extractor_overlapping_threshold,
                                       tmd::Config::dpm_detector_numthread);

        // apply clamp and make part coordinates relative to the box
        clamp_detections(frame->original_frame.cols,frame->original_frame.rows);

        std::vector<tmd::player_t *> players;
        for (tmd::detection detect : m_detections) {
            float score = std::get<2>(detect);
            cv::Rect box = std::get<0>(detect);
            std::vector<cv::Rect> parts = std::get<1>(detect);
            if (score > tmd::Config::dpm_extractor_score_threshold) {
                tmd::player_t *player = new player_t;
                player->frame_index = frame->frame_index;
                player->likelihood = score;
                player->mask_image = frame->mask_frame(box);
                player->original_image = frame->original_frame(box);
                player->pos_frame = box;
                player->features.body_parts = parts;
                extractTorsoForPlayer(player, std::get<3>(detect));
                players.push_back(player);
            }
        }
        m_detections.clear();
        return players;
    }

    void DPM::extractTorsoForPlayer(player_t *player, int component_level) {
        if (player == NULL) {
            throw std::invalid_argument("Error null pointer given to "
                                                "extractTorsoForPlayer method");
        }
        else if (player->features.body_parts.size() < 3) {
            throw std::invalid_argument("Error not enough body parts in "
                                                "extractTorsoForPlayer");
        }
        cv::Rect torso1;
        cv::Rect torso2;
        if (component_level == 1){
            torso1 = player->features.body_parts[1];
            torso2 = player->features.body_parts[2];
        }
        else{
            torso1 = player->features.body_parts[0];
            torso2 = player->features.body_parts[1];
        }
        cv::Rect mean;
        mean.x = (torso1.x + torso2.x) / 2;
        mean.y = (torso1.y + torso2.y) / 2;
        int oppoX = ((torso1.x + torso1.width) + (torso2.x + torso2.width)) / 2;
        int oppoY = ((torso1.y + torso1.height) + (torso2.y + torso2.height))/2;
        mean.width = oppoX - mean.x;
        mean.height = oppoY - mean.y;

        player->features.torso = (player->original_image.clone())(mean);
        player->features.torso_mask = (player->mask_image.clone())(mean);
        player->features.torso_pos = mean;
    }

    void DPM::cvLatentSvmDetectObjects(IplImage *image,
                                     CvLatentSvmDetector *detector,
                                     CvMemStorage *storage,
                                     float overlap_threshold, int numThreads) {
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
        getMaxFilterDims((const CvLSVMFilterObject **) (detector->filters),
                         detector->num_components, detector->num_part_filters,
                         &maxXBorder, &maxYBorder);
        // Create feature pyramid with nullable border
        H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
        // Search object
        error = this->searchObjectThresholdSomeComponents(H,
                          (const CvLSVMFilterObject **) (detector->filters),
                          detector->num_components, detector->num_part_filters,
                          detector->b, detector->score_threshold,
                          &points, &oppPoints, &score, &kPoints, numThreads);
        if (error != LATENT_SVM_OK) {
            return;
        }
        // Clipping boxes
        this->clippingBoxesLowerLeftCorner(image->width, image->height,
                                           oppPoints,
                                           kPoints);
        this->clippingBoxesUpperRightCorner(image->width, image->height, points,
                                            kPoints);
        // NMS procedure
        nonMaximumSuppression(kPoints, points, oppPoints, score,
                              overlap_threshold, &numBoxesOut, &pointsOut,
                              &oppPointsOut, &scoreOut);

        if (image->nChannels == 3)
            cvCvtColor(image, image, CV_RGB2BGR);

        freeFeaturePyramidObject(&H);
        free(points);
        free(oppPoints);
        free(score);
        free(scoreOut);
    }

    int DPM::clippingBoxesUpperRightCorner(int width, int height,
                                           CvPoint *points, int kPoints) {
        int i;
        for (i = 0; i < kPoints; i++) {
            cv::Rect &entry_i = std::get<0>(m_detections[i]);
            //assert(entry_i.x == points[i].x);
            //assert(entry_i.y == points[i].y);
            if (points[i].x > width - 1) {
                points[i].x = width - 1;
            }
            if (points[i].x < 0) {
                points[i].x = 0;
            }
            if (points[i].y > height - 1) {
                points[i].y = height - 1;
            }
            if (points[i].y < 0) {
                points[i].y = 0;
            }
            entry_i.x = points[i].x;
            entry_i.y = points[i].y;
        }
        return LATENT_SVM_OK;
    }

    int DPM::clippingBoxesLowerLeftCorner(int width, int height,
                                          CvPoint *points, int kPoints) {
        int i;
        for (i = 0; i < kPoints; i++) {
            CvPoint point = points[i];
            cv::Rect &entry_i = std::get<0>(m_detections[i]);
            if (points[i].x > width - 1) {
                points[i].x = width - 1;
            }
            if (points[i].x < 0) {
                points[i].x = 0;
            }
            if (points[i].y > height - 1) {
                points[i].y = height - 1;
            }
            if (points[i].y < 0) {
                points[i].y = 0;
            }
            entry_i.width = points[i].x - entry_i.x;
            entry_i.height = points[i].y - entry_i.y;
        }
        return LATENT_SVM_OK;
    }


    static void sort(int n, const float *x, int *indices) {
        int i, j;
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++) {
                if (x[indices[j]] > x[indices[i]]) {
                    //float x_tmp = x[i];
                    int index_tmp = indices[i];
                    //x[i] = x[j];
                    indices[i] = indices[j];
                    //x[j] = x_tmp;
                    indices[j] = index_tmp;
                }
            }
    }

    int DPM::nonMaximumSuppression(int numBoxes, const CvPoint *points,
                               const CvPoint *oppositePoints,
                               const float *score, float overlapThreshold,
                               int *numBoxesOut, CvPoint **pointsOut,
                               CvPoint **oppositePointsOut, float **scoreOut) {
        int i, j, index;
        float *box_area = (float *) malloc(numBoxes * sizeof(float));
        int *indices = (int *) malloc(numBoxes * sizeof(int));
        int *is_suppressed = (int *) malloc(numBoxes * sizeof(int));

        for (i = 0; i < numBoxes; i++) {
            indices[i] = i;
            is_suppressed[i] = 0;
            box_area[i] = (float) ((oppositePoints[i].x - points[i].x + 1) *
                                   (oppositePoints[i].y - points[i].y + 1));
        }

        sort(numBoxes, score, indices);
        for (i = 0; i < numBoxes; i++) {
            if (!is_suppressed[indices[i]]) {
                for (j = i + 1; j < numBoxes; j++) {
                    if (!is_suppressed[indices[j]]) {
                        int x1max = max(points[indices[i]].x,
                                        points[indices[j]].x);

                        int x2min = min(oppositePoints[indices[i]].x,
                                        oppositePoints[indices[j]].x);

                        int y1max = max(points[indices[i]].y,
                                        points[indices[j]].y);

                        int y2min = min(oppositePoints[indices[i]].y,
                                        oppositePoints[indices[j]].y);

                        int overlapWidth = x2min - x1max + 1;
                        int overlapHeight = y2min - y1max + 1;

                        if (overlapWidth > 0 && overlapHeight > 0) {
                            float overlapPart = (overlapWidth *
                                    overlapHeight) /  box_area[indices[j]];
                            if (overlapPart > overlapThreshold) {
                                is_suppressed[indices[j]] = 1;
                            }
                        }
                    }
                }
            }
        }

        *numBoxesOut = 0;
        for (i = 0; i < numBoxes; i++) {
            if (!is_suppressed[i]) (*numBoxesOut)++;
        }
        std::vector<tmd::detection> new_detections;
        *pointsOut = (CvPoint *) malloc((*numBoxesOut) * sizeof(CvPoint));
        *oppositePointsOut = (CvPoint *) malloc((*numBoxesOut)*sizeof(CvPoint));
        *scoreOut = (float *) malloc((*numBoxesOut) * sizeof(float));
        index = 0;
        for (i = 0; i < numBoxes; i++) {
            if (!is_suppressed[indices[i]]) {
                cv::Rect tmp = std::get<0>(m_detections[indices[i]]);
                (*pointsOut)[index].x = points[indices[i]].x;
                (*pointsOut)[index].y = points[indices[i]].y;
                (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
                (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
                (*scoreOut)[index] = score[indices[i]];
                new_detections.push_back(m_detections[indices[i]]);
                index++;
            }
        }
        m_detections = new_detections;
        free(indices);
        free(box_area);
        free(is_suppressed);

        return LATENT_SVM_OK;
    }


    std::vector<cv::Rect> DPM::get_parts_rect_for_point(const
                                                CvLSVMFilterObject **filters,
                                                int n, CvPoint
                                                *partsDisplacement, int
                                                level) {
        int j;
        float step;
        CvPoint oppositePoint;
        std::vector<cv::Rect> parts;

        step = powf(2.0f, 1.0f / ((float) LAMBDA));
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

    int DPM::searchObjectThresholdSomeComponents(const
                                     CvLSVMFeaturePyramid *H,
                                     const CvLSVMFilterObject **filters,
                                     int kComponents, const int *kPartFilters,
                                     const float *b, float scoreThreshold,
                                     CvPoint **points, CvPoint **oppPoints,
                                     float **score, int *kPoints,
                                     int numThreads) {
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
        partsDisplacementArr = (CvPoint ***) malloc(sizeof(CvPoint **)  *
                                                            kComponents);

        // Getting maximum filter dimensions
        /*error = */getMaxFilterDims(filters, kComponents, kPartFilters,
                                     &maxXBorder, &maxYBorder);
        componentIndex = 0;
        *kPoints = 0;
        // For each component perform searching
        for (i = 0; i < kComponents; i++) {

            int error = this->searchObjectThreshold(H,
                        &(filters[componentIndex]), kPartFilters[i],
                        b[i], maxXBorder, maxYBorder, scoreThreshold,
                        &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
                        &(scoreArr[i]), &(partsDisplacementArr[i]), numThreads);

            if (error != LATENT_SVM_OK) {
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
                                filters[componentIndex]->sizeX,
                                filters[componentIndex]->sizeY,
                                &(oppPointsArr[i]));

            componentIndex += (kPartFilters[i] + 1);
            *kPoints += kPointsArr[i];
        }

        *points = (CvPoint *) malloc(sizeof(CvPoint) * (*kPoints));
        *oppPoints = (CvPoint *) malloc(sizeof(CvPoint) * (*kPoints));
        *score = (float *) malloc(sizeof(float) * (*kPoints));
        s = 0;
        for (i = 0; i < kComponents; i++) {
            f = s + kPointsArr[i];
            for (j = s; j < f; j++) {
                (*points)[j].x = pointsArr[i][j - s].x;
                (*points)[j].y = pointsArr[i][j - s].y;
                (*oppPoints)[j].x = oppPointsArr[i][j - s].x;
                (*oppPoints)[j].y = oppPointsArr[i][j - s].y;
                (*score)[j] = scoreArr[i][j - s];
                std::vector<cv::Rect> p = get_parts_rect_for_point
                        (filters, kPartFilters[i],
                         partsDisplacementArr[i][j - s], levelsArr[i][j - s]);
                tmd::detection entry = std::make_tuple(cv::Rect((*points)[j],
                                        (*oppPoints)[j]), p, (*score)[j], i);
                m_detections.push_back(entry);
            }
            s = f;
        }

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
        return LATENT_SVM_OK;
    }

    int DPM::estimateBoxes(CvPoint *points, int *levels, int kPoints,
                           int sizeX, int sizeY, CvPoint **oppositePoints) {
        int i;
        float step;

        step = powf(2.0f, 1.0f / ((float) (LAMBDA)));

        *oppositePoints = (CvPoint *) malloc(sizeof(CvPoint) * kPoints);
        for (i = 0; i < kPoints; i++) {
            this->getOppositePoint(points[i], sizeX, sizeY, step, levels[i] -
                                              LAMBDA, &((*oppositePoints)[i]));
        }
        return LATENT_SVM_OK;
    }

    int DPM::getOppositePoint(CvPoint point,
                              int sizeX, int sizeY,
                              float step, int degree,
                              CvPoint *oppositePoint) {
        float scale;
        scale = SIDE_LENGTH * powf(step, (float) degree);
        oppositePoint->x = (int) (point.x + sizeX * scale);
        oppositePoint->y = (int) (point.y + sizeY * scale);
        return LATENT_SVM_OK;
    }

    int DPM::searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                                   const CvLSVMFilterObject **all_F, int n,
                                   float b,
                                   int maxXBorder, int maxYBorder,
                                   float scoreThreshold,
                                   CvPoint **points, int **levels, int *kPoints,
                                   float **score, CvPoint ***partsDisplacement,
                                   int numThreads) {
        int opResult;

        if (numThreads <= 0)
    {
        opResult = LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT;
        return opResult;
    }
    opResult = tbbThresholdFunctionalScore(all_F, n, H, b,maxXBorder,maxYBorder,
                                           scoreThreshold, numThreads, score,
                                           points, levels, kPoints,
                                           partsDisplacement);

        if (opResult != LATENT_SVM_OK) {
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

    int DPM::convertPoints(int /*countLevel*/, int lambda,
                           int initialImageLevel,
                           CvPoint *points, int *levels,
                           CvPoint **partsDisplacement, int kPoints, int n,
                           int maxXBorder,
                           int maxYBorder) {
        int i, j, bx, by;
        float step, scale;
        step = powf(2.0f, 1.0f / ((float) lambda));

        computeBorderSize(maxXBorder, maxYBorder, &bx, &by);

        for (i = 0; i < kPoints; i++) {
            // scaling factor for root filter
            scale = SIDE_LENGTH*powf(step,(float)(levels[i]-initialImageLevel));
            points[i].x = (int) ((points[i].x - bx + 1) * scale);
            points[i].y = (int) ((points[i].y - by + 1) * scale);

            // scaling factor for part filters
            scale = SIDE_LENGTH * powf(step,
                           (float) (levels[i] - lambda - initialImageLevel));

            for (j = 0; j < n; j++) {
                partsDisplacement[i][j].x = (int) ((partsDisplacement[i][j].x -
                                                    2 * bx + 1) * scale);
                partsDisplacement[i][j].y = (int) ((partsDisplacement[i][j].y -
                                                    2 * by + 1) * scale);
            }
        }
        return LATENT_SVM_OK;
    }

    void DPM::clamp_detections(int width, int height) {
        for (tmd::detection &detect : m_detections) {
            cv::Rect &box = std::get<0>(detect);
            box.x = max (0, box.x);
            box.y = max (0, box.y);
            box.width = min(box.width, width);
            box.height = min(box.height, height);
            for (cv::Rect &part : std::get<1>(detect)) {
                part.x -= box.x;
                part.y -= box.y;
                if (part.x < 0) {
                    part.x = 0;
                }
                if (part.y < 0) {
                    part.y = 0;
                }
                if (part.x > box.width){
                    part.x = box.width;
                }
                if (part.y > box.height){
                    part.y = box.height;
                }
                if (part.width < 0){
                    part.width = 0;
                }
                if (part.height < 0){
                    part.height = 0;
                }
                if (part.x + part.width > box.width) {
                    part.width = box.width - part.x;
                }
                if (part.y + part.height > box.height) {
                    part.height = box.height - part.y;
                }
            }
        }
    }
}