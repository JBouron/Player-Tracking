#include <opencv2/core/core.hpp>
#include "../headers/feature_comparator.h"
#include <iostream>
#include <fstream>
#include <bits/stream_iterator.h>


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

        double min = distances.at<double>(0);
        int minIndex = 0;
        for(int i = 1; i < distances.rows; i ++){
            if(distances.at<double>(i) < min){
                min = distances.at<double>(i,0);
                minIndex = i;
            }
        }

        return m_centers.at<Mat>(minIndex);
    }

    cv::Mat FeatureComparator::getClosestCenter(player_t *player, int i) {
        return getClosestCenter(getMatForPlayerFeature(player, i));
    }

    void FeatureComparator::addPlayerFeatures(player_t *player, int i) {
        Mat meanAsMat = getMatForPlayerFeature(player,i);
        addSampleToData(meanAsMat);
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

    cv::Mat FeatureComparator::getMatForPlayerFeature(player_t *player, int i) {
        Rect rect = player->features.body_parts[i];
        Mat dpmPartMask = player->mask_image(rect);
        Mat dpmPart = player->original_image(rect);

        Scalar stripMean = mean(dpmPart, dpmPartMask);
        //ASSUMING 3 CHANNEL STRIP
        double meanChannel1 = stripMean[0];
        double meanChannel2 = stripMean[1];
        double meanChannel3 = stripMean[2];
        Mat meanAsMat(1, 3, CV_64F);
        meanAsMat.at<double>(0, 0) = meanChannel1;
        meanAsMat.at<double>(0, 1) = meanChannel2;
        meanAsMat.at<double>(0, 2) = meanChannel3;
        return meanAsMat;
    }

    void FeatureComparator::writeCentersToFile() {
        std::ofstream clustersFile ("clusterCenters.txt");
        if (clustersFile.is_open())
        {
            for(int i = 0; i < m_centers.rows; i++)
            {
                for(int j = 0; j < m_centers.cols; j++)
                {
                    if(j < m_centers.cols -1)
                    {
                        clustersFile << m_centers.at(i,j) << " ";
                    }
                    else{
                        clustersFile << m_centers.at(i,j);
                    }

                }
                clustersFile <<"\n";
            }
            clustersFile.close();
        }
    }

    Mat FeatureComparator::readCentersFromFile() {
        std::ofstream clustersFile ("clusterCenters.txt");
        Mat toReturn;
        if(clustersFile.is_open())
        {
            string line;
            while(getline(clustersFile, line))
            {
                std::stringstream ss(line);
                std::istream_iterator<std::string> begin(ss);
                std::istream_iterator<std::string> end;
                std::vector<std::string> vstrings(begin, end);
                std::vector<double> doubleVector(vstrings.size());
                std::transform(vstrings.begin(), vstrings.end(), doubleVector.begin(), [](const std::string& val)
                {
                    return std::stod(val);
                });
                toReturn.push_back(doubleVector);
            }
        }
    }
}