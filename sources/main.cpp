#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../headers/heuristic_feature_extractor.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"
#include "../headers/frame_t.h"
#include "../headers/debug.h"
#include "../headers/bgsubstractor.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/frame_t.h"

#include "/home/jbouron/openCV-2.4.11/opencv-2.4.11/modules/objdetect/src/_latentsvm.h"
#include "../headers/dpm_detector.h"


#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#include <opencv2/imgproc/imgproc_c.h>

#endif



using namespace std;
using namespace cv;

static void help()
{
    cout << "This program demonstrated the use of the latentSVM detector." << endl <<
    "It reads in trained object models and then uses them to detect the objects in images." << endl <<
    endl <<
    "Call:" << endl <<
    "./latentsvm_multidetect <imagesFolder> <modelsFolder> [<overlapThreshold>][<threadsNumber>]" << endl <<
    "<overlapThreshold> - threshold for the non-maximum suppression algorithm." << endl <<
    "Example of <modelsFolder> is opencv_extra/testdata/cv/latentsvmdetector/models_VOC2007" << endl <<
    endl <<
    "Keys:" << endl <<
    "'n' - to go to the next image;" << endl <<
    "'esc' - to quit." << endl <<
    endl;
}

static void detectAndDrawObjects( Mat& image, LatentSvmDetector& detector, const vector<Scalar>& colors, float overlapThreshold, int numThreads )
{
    vector<LatentSvmDetector::ObjectDetection> detections;

    TickMeter tm;
    tm.start();
    detector.detect( image, detections, overlapThreshold, numThreads);
    tm.stop();

    cout << "Detection time = " << tm.getTimeSec() << " sec" << endl;

    const vector<string> classNames = detector.getClassNames();
    CV_Assert( colors.size() == classNames.size() );

    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& od = detections[i];
        if (od.score > -10.0)
            rectangle( image, od.rect, colors[od.classID], 3 );
    }
    // put text over the all rectangles
    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& od = detections[i];
        putText( image, classNames[od.classID], Point(od.rect.x+4,od.rect.y+13), FONT_HERSHEY_SIMPLEX, 0.55, colors[od.classID], 2 );
    }
}

static void readDirectory( const string& directoryName, vector<string>& filenames, bool addDirectoryName=true )
{
    filenames.clear();

#if defined(WIN32) | defined(_WIN32)
    struct _finddata_t s_file;
    string str = directoryName + "\\*.*";

    intptr_t h_file = _findfirst( str.c_str(), &s_file );
    if( h_file != static_cast<intptr_t>(-1.0) )
    {
        do
        {
            if( addDirectoryName )
                filenames.push_back(directoryName + "\\" + s_file.name);
            else
                filenames.push_back((string)s_file.name);
        }
        while( _findnext( h_file, &s_file ) == 0 );
    }
    _findclose( h_file );
#else
    DIR* dir = opendir( directoryName.c_str() );
    if( dir != NULL )
    {
        struct dirent* dent;
        while( (dent = readdir(dir)) != NULL )
        {
            if( addDirectoryName )
                filenames.push_back( directoryName + "/" + string(dent->d_name) );
            else
                filenames.push_back( string(dent->d_name) );
        }

        closedir( dir );
    }
#endif

    sort( filenames.begin(), filenames.end() );
}

void heuristic_features_extractor_tests() {
    tmd::player_t p;
    p.original_image = (imread(
            "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/test_image.jpg"));
    tmd::HeuristicFeaturesExtractor d;
            d.extract_features(&p);
    cv::namedWindow("Strips");
    for (int i = 0; i < p.features.strips.size(); i ++){
        imshow("Strips", p.features.strips[i]);
        waitKey(0);
    }
}
void bgs_demo(){
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
    VideoCapture* capture = new VideoCapture("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/videos/ace_1.mp4");
    tmd::BGSubstractor bgs(capture, 1);
    while (bgs.has_next_frame()){
        tmd::debug("New iteration");
        tmd::frame_t* frame = bgs.next_frame();
        imshow("Frame", frame->original_frame);
        imshow("FG Mask MOG 2", frame->mask_frame);
        frame->original_frame.release();
        frame->mask_frame.release();
        free(frame);
        cv::waitKey(1);
    }
        tmd::debug("End");
}

void manual_player_extractor_test(){
    tmd::frame_t frame;
    (frame.original_frame) = (imread(
            "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/TEASER-Basketball-bomb.jpg"));
    tmd::ManualPlayerExtractor pe;
    std::vector<tmd::player_t*> v = pe.extract_player_from_frame(&frame);
    tmd::HeuristicFeaturesExtractor d;
    d.extract_features_from_players(v);
    namedWindow("Features");
    for (size_t i = 0 ; i < v.size(); i ++){
        std::vector<cv::Mat> strips = v[i]->features.strips;
        for (size_t j = 0 ; j < strips.size() ; j ++){
            cv::imshow("Features", strips[j]);
            cv::waitKey(0);
        }
    }
}

int min_x_filters(CvLatentSvmDetector* detector){
    CvLSVMFilterObject** filters = detector->filters;
    const int filter_count = detector->num_filters;
    int min = 0;
    for (int i = 0 ; i < filter_count ; i ++){
        int actual = filters[i]->sizeX;
        if (i == 0) min = actual;
        else{
            if (actual < min){
                min = actual;
            }
        }
    }
    return min;
}

int min_y_filters(CvLatentSvmDetector* detector){
    CvLSVMFilterObject** filters = detector->filters;
    const int filter_count = detector->num_filters;
    int min = 0;
    for (int i = 0 ; i < filter_count ; i ++){
        int actual = filters[i]->sizeY;
        if (i == 0) min = actual;
        else{
            if (actual < min){
                min = actual;
            }
        }
    }
    return min;
}
/*
//DONE
static int estimateBoxes(CvPoint *points, int *levels, int kPoints,
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

//DONE
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters,
                     unsigned int *maxXBorder, unsigned int *maxYBorder)
{
    int i, componentIndex;
    *maxXBorder = filters[0]->sizeX;
    *maxYBorder = filters[0]->sizeY;
    componentIndex = kPartFilters[0] + 1;
    for (i = 1; i < kComponents; i++)
    {
        if ((unsigned)filters[componentIndex]->sizeX > *maxXBorder)
        {
            *maxXBorder = filters[componentIndex]->sizeX;
        }
        if ((unsigned)filters[componentIndex]->sizeY > *maxYBorder)
        {
            *maxYBorder = filters[componentIndex]->sizeY;
        }
        componentIndex += (kPartFilters[i] + 1);
    }
    return LATENT_SVM_OK;
}


//DONE
int CustomshowPartFilterBoxes(IplImage *image,
                        const CvLSVMFilterObject **filters,
                        int n, CvPoint **partsDisplacement,
                        int *levels, int kPoints,
                        CvScalar color, int thickness,
                        int line_type, int shift, float* scores)
{
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

//DONE
int CustomsearchObjectThresholdSomeComponents(IplImage* image, const CvLSVMFeaturePyramid *H,
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
    getMaxFilterDims(filters, kComponents, kPartFilters, &maxXBorder, &maxYBorder);
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
        estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
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
    if (draw) CustomShowPartFilterBoxes(image, filters,
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

//DONE
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

//DONE
int nonMaximumSuppression(int numBoxes, const CvPoint *points,
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

//DONE
CvSeq* CustomcvLatentSvmDetectObjects(IplImage* image,
                                CvLatentSvmDetector* detector,
                                CvMemStorage* storage,
                                float overlap_threshold, int numThreads)
{
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
    error = CustomSearchObjectThresholdSomeComponents(image, H, (const CvLSVMFilterObject**)(detector->filters),
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
}*/

void test_dpm_class(){
    tmd::DPMDetector d("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/xmls/person.xml", 0.5f, 1);
    IplImage* image = cvLoadImage("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/img3.jpg");
    d.testOnImage(image);
}


void show_root_boxes(IplImage* image, CvLatentSvmDetector* detector, int index){
    if (index >= detector->num_filters){
        return;
    }
    const int point_count = 1;
    const int level_count = 1;
    CvLSVMFilterObject* filter = detector->filters[index];
    CvPoint* points_set = new CvPoint[point_count];
    points_set[0].x = filter->V.x;
    points_set[0].y = filter->V.y;
    int* levels = new int[level_count];
    levels[0] = filter->V.l+1;
    CvScalar color;
    color.val[0] = ((double)index / detector->num_filters)*255; color.val[1] = (1 - ((double)index / detector->num_filters))*255; color.val[2] = 255; color.val[3] = 255;
    const int thickness = 1;
    const int line_type = 8; // 8 connected line.
    const int shift = 0;
    tmd::debug("level = " + std::to_string(levels[0]));
    showRootFilterBoxes(image, filter, points_set, levels, point_count, color, thickness, line_type, shift);
    waitKey(0);
    show_root_boxes(image, detector, index + 1);
}

void show_part_boxes(IplImage* image, CvLatentSvmDetector* detector){
    const CvLSVMFilterObject** filters = (const CvLSVMFilterObject**) detector->filters;
    int part_filter_count = detector->num_filters;
    CvPoint** parts_displacement = new CvPoint*[part_filter_count];
    int* levels = new int[part_filter_count];
    for (int i = 0; i < part_filter_count ; i ++){
        parts_displacement[i] = new CvPoint;
        parts_displacement[i]->x = filters[i]->V.x;
        parts_displacement[i]->y = filters[i]->V.y;
        levels[i] = filters[i]->V.l;
    }
    CvScalar color;
    color.val[0] = 0; color.val[1] = 255; color.val[2] = 255; color.val[3] = 255;
    const int thickness = 1;
    const int line_type = 8; // 8 connected line.
    const int shift = 0;
    showPartFilterBoxes(image, filters, part_filter_count, parts_displacement, levels, part_filter_count, color, thickness, line_type, shift);
}

void lsvm_c(){
    CvLatentSvmDetector* d = cvLoadLatentSvmDetector("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/xmls/person.xml");
    CvMemStorage* memStorage = cvCreateMemStorage(0);
    IplImage* imageTOPKEK = cvLoadImage("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/img3.jpg");
    CvSeq* seq = NULL;
    tmd::debug("score threshold = "  + std::to_string(d->score_threshold));
    //show_root_boxes(imageTOPKEK, d, 0);
    //seq = CustomcvLatentSvmDetectObjects(imageTOPKEK, d, memStorage, 0.5f, 1);

    CvObjectDetection* obj = new CvObjectDetection;

    cvSeqPop(seq, obj);
    if (seq == NULL){
        tmd::debug("NULL pointer");
    }
    tmd::debug("rect = " + std::to_string(obj->rect.x) + "  " + std::to_string(obj->rect.y) + "  " + std::to_string(obj->rect.width) + "  " + std::to_string(obj->rect.height));
    tmd::debug("seq size = " + std::to_string(seq->total));
    for(int i = 0; i < seq->total; i++ )
    {
        CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( seq, i );
        float score         = detection.score;
        CvRect bounding_box = detection.rect;
        cvRectangle( imageTOPKEK, cvPoint(bounding_box.x, bounding_box.y),
                     cvPoint(bounding_box.x + bounding_box.width,
                             bounding_box.y + bounding_box.height),
                     CV_RGB(cvRound(255.0f*score),0,0), 3 );
    }
    //show_root_boxes(imageTOPKEK, d, 0);
    waitKey(0);
    cvReleaseLatentSvmDetector(&d);
}

int main(int argc, char* argv[]) {
    test_dpm_class();
    return 0;

}