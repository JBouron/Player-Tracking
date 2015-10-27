#include <opencv2/core/core.hpp>
#include "../headers/feature_comparator.h"
#include "../headers/features_t.h"

using namespace cv;

namespace tmd{

    FeatureComparator::FeatureComparator(cv::Mat data, int clusterCount, cv::Mat labels, cv::TermCriteria criteria,
                                         int attempts, int flags, cv::Mat centers) {
        m_data = data;
        m_clusterCount = clusterCount;
        m_labels = labels;
        m_termCriteria = criteria;
        m_attempts = attempts;
        m_flags = flags;
        m_centers = centers;
    }

    FeatureComparator::~FeatureComparator(){
        m_data.release();
        m_labels.release();
        m_centers.release();
    }

    void FeatureComparator::runClustering() {
        kmeans(m_data, m_clusterCount,m_labels, m_termCriteria, m_attempts, m_flags, m_centers);
    }

    void FeatureComparator::addSampleToData(cv::Mat sample) {
        if(m_data.cols != sample.cols || sample.rows != 1) {
            throw std::invalid_argument("Sample doesn't have the same amount of dimensions as the data !");
        }
        m_data.push_back(sample);
    }

    cv::Mat FeatureComparator::getClosestCenter(cv::Mat sample) {
        if(m_data.cols != sample.cols){
            throw std::invalid_argument("Sample doesn't have the same amount of dimensions as the data !");
        }

        Mat distances(m_centers.rows, 1, CV_64F);

        for(int i = 0; i < m_centers.rows; i ++){
            distances.push_back(norm(m_centers.row(i), sample, NORM_L2));
        }

        double min =  distances.at(0);
        int minIndex = 0;
        for(int i = 1; i < distances.rows; i ++){
            if(distances.at(i) < min){
                min = distances.at(i);
                minIndex = i;
            }
        }

        return m_centers.at(minIndex);
    }

    cv::Mat FeatureComparator::getClosestCenter(features_t feature, int i) {
        if (feature.strips[i].channels() != 3) {
            throw std::invalid_argument("Feature strips don't have 3 channels !");
        }
        Scalar stripMean = mean(feature.strips[i]);
        //ASSUMING 3 CHANNEL STRIP
        double meanChannel1 = stripMean[0];
        double meanChannel2 = stripMean[1];
        double meanChannel3 = stripMean[2];
        Mat meanAsMat(1, 3, CV_64F);
        meanAsMat.at(0, 0) = meanChannel1;
        meanAsMat.at(0, 1) = meanChannel2;
        meanAsMat.at(0, 2) = meanChannel3;
        return getClosestCenter(meanAsMat);
    }

    void FeatureComparator::addPlayerFeatures(features_t feature, int i) {
        if(feature.strips[i].channels() != 3){
            throw std::invalid_argument("Feature strips don't have 3 channels !");
        }
        Mat meanAsMat = getMatForFeature(feature,i);
        addSampleToData(meanAsMat);

    }

    cv::Mat FeatureComparator::getMatForFeature(features_t feature, int i) {
        Scalar stripMean = mean(feature.strips[i]);
        //ASSUMING 3 CHANNEL STRIP
        double meanChannel1 = stripMean[0];
        double meanChannel2 = stripMean[1];
        double meanChannel3 = stripMean[2];
        Mat meanAsMat(1, 3, CV_64F);
        meanAsMat.at(0, 0) = meanChannel1;
        meanAsMat.at(0, 1) = meanChannel2;
        meanAsMat.at(0, 2) = meanChannel3;
        return meanAsMat;
    }

    void FeatureComparator::setTermCriteria(cv::TermCriteria criteria) {
        m_termCriteria = criteria;
    }

    void FeatureComparator::setAttempts(int attempts) {
        m_attempts = attempts;
    }

    void FeatureComparator::setFlags(int flags) {
        m_flags = flags;
    }
}