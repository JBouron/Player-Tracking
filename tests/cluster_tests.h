#ifndef BACHELOR_PROJECT_CLUSTER_TESTS_H
#define BACHELOR_PROJECT_CLUSTER_TESTS_H

#include "../headers/dpm_detector.h"
#include "../headers/feature_comparator.h"
#include "dpm_detector_tests.h"

using namespace cv;

namespace tmd{
    void manual_player_comparator_test() {
        tmd::DPMDetector d(
                "/home/nicolas/Documents/EPFL/Projet/Code/Bachelor-Project/res/xmls/person.xml");
        std::vector<tmd::player_t *> v = tmd::get_player_vector();
        Mat data(0, 3, CV_32F), labels;
        Mat centers;
        tmd::FeatureComparator comparator(data, 3, labels, TermCriteria(
                                                  CV_TERMCRIT_EPS +
                                                  CV_TERMCRIT_ITER, 10, 1.0), 3,
                                          KMEANS_PP_CENTERS, centers);

        for (int i = 0; i < v.size(); i++) {
            d.extractBodyParts(v[i]);
            comparator.addPlayerFeatures(v[i]);
        }

        comparator.runClustering();
        comparator.writeCentersToFile();
        Mat readCenters = comparator.readCentersFromFile(3, 180);
        comparator.~FeatureComparator();
    }
}
#endif //BACHELOR_PROJECT_CLUSTER_TESTS_H
