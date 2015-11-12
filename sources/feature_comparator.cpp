#include <opencv2/core/core.hpp>
#include "../headers/feature_comparator.h"
#include <iostream>
#include <fstream>
#include <bits/stream_iterator.h>

using namespace cv;

namespace tmd {

    FeatureComparator::FeatureComparator(cv::Mat data, int clusterCount,
                                         cv::Mat labels,
                                         cv::TermCriteria criteria,
                                         int attempts, int flags,
                                         cv::Mat centers) {
        m_data = data;
        m_clusterCount = clusterCount;
        m_labels = labels;
        m_termCriteria = criteria;
        m_attempts = attempts;
        m_flags = flags;
        m_centers = centers;
    }

    FeatureComparator::~FeatureComparator() {
        m_data.release();
        m_labels.release();
        m_centers.release();
    }

    void FeatureComparator::runClustering() {
        m_labels = m_data;
        kmeans(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
               m_flags, m_centers);
    }

    void FeatureComparator::addSampleToData(cv::Mat sample) {
        if (m_data.cols != sample.cols || sample.rows != 1) {
            throw std::invalid_argument(
                    "Sample doesn't have the same amount of dimensions as the data !");
        }
        m_data.push_back(sample);
    }

    cv::Mat FeatureComparator::getClosestCenter(cv::Mat sample) {
        if (m_data.cols != sample.cols) {
            throw std::invalid_argument(
                    "Sample doesn't have the same amount of dimensions as the data !");
        }

        Mat distances(m_centers.rows, 1, CV_32F);

        for (int i = 0; i < m_centers.rows; i++) {
            float distance = (float) norm(m_centers.row(i), sample,
                                                   NORM_L2);
            distances.at<float>(i,0) = distance;
        }

        float min = distances.at<float>(0);
        int minIndex = 0;
        for (int i = 1; i < distances.rows; i++) {
            if (distances.at<float>(i) < min) {
                min = distances.at<float>(i, 0);
                minIndex = i;
            }
        }

        return m_centers.row(minIndex);
    }

    cv::Mat FeatureComparator::getClosestCenter(player_t *player) {
        return getClosestCenter(getMatForPlayerFeature(player));
    }

    void FeatureComparator::addPlayerFeatures(player_t *player) {
        Mat meanAsMat = getMatForPlayerFeature(player);
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

    cv::Mat FeatureComparator::getMatForPlayerFeature(player_t *player) {
        return player->features.torso_color_histogram;
    }

    void FeatureComparator::writeCentersToFile() {
        std::ofstream clustersFile("/home/nicolas/Desktop/clusterCenters.txt");
        if (clustersFile.is_open()) {
            for (int i = 0; i < m_centers.rows; i++) {
                for (int j = 0; j < m_centers.cols; j++) {
                    Mat row = m_centers.row(i);
                    if (j < m_centers.cols - 1) {
                        clustersFile << row.at<float>(j) << " ";
                    }
                    else {
                        clustersFile << row.at<float>(j);
                    }
                }
                clustersFile << "\n";
            }
            clustersFile.close();
        }
    }

    Mat FeatureComparator::readCentersFromFile(int rows, int cols) {
        std::ifstream clustersFile("/home/nicolas/Desktop/clusterCenters.txt");
        Mat toReturn(rows, cols, CV_32F);
        if (clustersFile.is_open()) {
            string line;
            for(int j = 0; j < rows; j++){
                getline(clustersFile, line);
                vector<float> floatVector = getFloatsFromString(line);
                for(int i = 0; i < cols; i ++){
                    toReturn.at<float>(j, i) = floatVector[i];
                }
            }
            clustersFile.close();
        }
        return toReturn;
    }

    std::vector<float> FeatureComparator::getFloatsFromString(
            std::string inputString) {
        std::stringstream ss(inputString);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> vstrings(begin, end);
        std::vector<float> floatVector(vstrings.size());
        std::transform(vstrings.begin(), vstrings.end(), floatVector.begin(),
                       [](const std::string &val) {
                           return std::stof(val);
                       });
        return floatVector;
    }

    cv::Mat FeatureComparator::getData() {
        return m_data;
    }
}