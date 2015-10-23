#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../headers/heuristic_feature_extractor.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"
#include "../headers/manual_player_extractor.h"
#include "../headers/frame_t.h"

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
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

void manual_player_extractor_test(){
    tmd::frame_t frame;
    (frame.original_frame) = new cv::Mat();
    *(frame.original_frame) = (imread(
            "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/TEASER-Basketball-bomb.jpg"));
    tmd::ManualPlayerExtractor pe;
    pe.extract_player_from_frame(&frame);
}

int main(int argc, char* argv[])
{
    manual_player_extractor_test();
    return EXIT_SUCCESS;
    help();
    string images_folder, models_folder;
    float overlapThreshold = 0.2f;
    int numThreads = -1;
    if( argc > 2 )
    {
        images_folder = argv[1];
        models_folder = argv[2];
        if( argc > 3 ) overlapThreshold = (float)atof(argv[3]);
        if( overlapThreshold < 0 || overlapThreshold > 1)
        {
            cout << "overlapThreshold must be in interval (0,1)." << endl;
            exit(-1);
        }

        if( argc > 4 ) numThreads = atoi(argv[4]);
    }

    vector<string> images_filenames, models_filenames;
    readDirectory( images_folder, images_filenames );
    readDirectory( models_folder, models_filenames );

    LatentSvmDetector detector( models_filenames );
    if( detector.empty() )
    {
        cout << "Models can't be loaded" << endl;
        exit(-1);
    }

    const vector<string>& classNames = detector.getClassNames();
    cout << "Loaded " << classNames.size() << " models:" << endl;
    for( size_t i = 0; i < classNames.size(); i++ )
    {
        cout << i << ") " << classNames[i] << "; ";
    }
    cout << endl;

    cout << "overlapThreshold = " << overlapThreshold << endl;

    vector<Scalar> colors;
    generateColors( colors, detector.getClassNames().size() );

    for( size_t i = 0; i < images_filenames.size(); i++ )
    {
        Mat image = imread( images_filenames[i] );
        if( image.empty() )  continue;

        cout << "Process image " << images_filenames[i] << endl;
        detectAndDrawObjects( image, detector, colors, overlapThreshold, numThreads );

        imshow( "result", image );

        for(;;)
        {
            int c = waitKey();
            if( (char)c == 'n')
                break;
            else if( (char)c == '\x1b' )
                exit(0);
        }
    }

    return 0;
}
