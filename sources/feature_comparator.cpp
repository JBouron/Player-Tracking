#include <opencv2/core/core.hpp>
#include "../headers/feature_comparator.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"
#include "../headers/debug.h"
#include <iostream>
#include <fstream>
#include <bits/stream_iterator.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace tmd {

    FeatureComparator::FeatureComparator(cv::Mat data, int clusterCount,
                                         cv::Mat labels,
                                         cv::TermCriteria criteria,
                                         int attempts, int flags,
                                         cv::Mat centers) {
        if (data.rows == 0 || data.cols == 0) {
            throw std::invalid_argument("Data cannot be empty !");
        }
        m_data = data;
        m_sampleCols = m_data.cols;
        m_clusterCount = clusterCount;
        m_labels = labels;
        m_termCriteria = criteria;
        m_attempts = attempts;
        m_flags = flags;
        m_centers = centers;
        computeColorCentersIndexes();
    }

    FeatureComparator::FeatureComparator(int clusterCount, int sampleCols,
                                         cv::Mat centers) {
        m_data = cv::Mat(0, sampleCols, CV_32F);
        m_clusterCount = clusterCount;
        m_labels = cv::Mat(m_data);
        m_termCriteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                                          10, 1.0);;
        m_attempts = 3;
        m_flags = cv::KMEANS_PP_CENTERS;
        m_centers = centers;
        m_sampleCols = sampleCols;
        computeColorCentersIndexes();
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
        computeColorCentersIndexes();
    }

    void FeatureComparator::addSampleToData(cv::Mat sample) {
        if (m_sampleCols != sample.cols || sample.rows != 1) {
            throw std::invalid_argument(
                    "Sample doesn't have the same amount of dimensions as the data !");
        }

        m_data.push_back(sample);
    }

    double FeatureComparator::getClosestCenter(cv::Mat sample) {
        if (m_data.cols != sample.cols || sample.rows != 1) {
            throw std::invalid_argument(
                    "Sample doesn't have the same amount of dimensions as the data !");
        }

        double distances[m_centers.rows];
        double max = 0;
        int index_max = -1;
        for (int i = 0; i < m_centers.rows; i++) {
            distances[i] = compareHist(m_centers.row(i), sample, CV_COMP_CORREL);
            std::cout << "Correlation with center = " << distances[i] << std::endl;
            std::cout << "Center index = " << i << std::endl;
            if(distances[i] > max){
                max = distances[i];
                index_max = i;
            }
        }
        std::cout << "max = " << max << std::endl;
        return index_max;

        /*

        float max_diff_area = 0.0;
        int max_area_center = 0;
        for (int i = 0; i < m_centers.rows; i++) {
            float area = 0;
            for (int h = 0; h < m_centers.row(i).cols; h++) {
                area += MIN(m_centers.at<float>(i, h), sample.at<float>(0, h));
            }
            std::cout << "Area with center " << i << " = " << area <<
            std::endl;
            if (area > max_diff_area) {
                max_diff_area = area;
                max_area_center = i;
            }
        }

        return max_area_center;
         */
    }

    double FeatureComparator::getClosestCenter(player_t *player) {
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
        cv::Mat t;
        cv::transpose(player->features.torso_color_histogram, t);
        return t;
    }

    void FeatureComparator::writeCentersToFile() {
        std::ofstream clustersFile("./res/cluster/clusterCenters.txt");
        std::cout << "Write centers " << std::endl;
        if (clustersFile.is_open()) {
            for (int i = 0; i < m_centers.rows; i++) {
                for (int j = 0; j < m_centers.cols; j++) {
                    Mat row = m_centers.row(i);
                    std::cout << row.at<float>(j) << ", ";
                    if (j < m_centers.cols - 1) {
                        clustersFile << row.at<float>(j) << " ";
                    }
                    else {
                        clustersFile << row.at<float>(j);
                    }
                }
                std::cout << std::endl;
                clustersFile << "\n";
            }
            clustersFile.flush();
            clustersFile.close();
        }
    }

    Mat FeatureComparator::readCentersFromFile(int rows, int cols) {
        std::cout << "readCentersFromFile :" << std::endl;
        std::ifstream clustersFile("./res/cluster/clusterCenters.txt");
        if (!clustersFile.is_open()) {
            throw std::runtime_error("Error couldn't load clusterCenters.txt");
        }
        Mat toReturn(rows, cols, CV_32F);
        if (clustersFile.is_open()) {
            string line;
            for (int j = 0; j < rows; j++) {
                getline(clustersFile, line);
                vector<float> floatVector = getFloatsFromString(line);
                for (int i = 0; i < cols; i++) {
                    std::cout << floatVector[i] << ", ";
                    toReturn.at<float>(j, i) = floatVector[i];
                }
                std::cout << std::endl;
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

    void FeatureComparator::computeColorCentersIndexes() {
        if (m_centers.rows == 0) {
            return;
        }

        /*float max_value = 0;
        int max_hue = 0;
        for (int i = 0 ; i < m_centers.row(0).cols ; i ++) {
            if (m_centers.at<float>(0, i) > max_value) {
                max_value = m_centers.at<float>(0, i);
                max_hue = i;
            }
        }*/

        float mean0 = 0;
        float total0 = 0;
        for (int i = 0; i < m_centers.row(0).cols; i++) {
            mean0 += m_centers.at<float>(0, i) * i;
            total0 += m_centers.at<float>(0, i);
        }
        mean0 = mean0 / total0;

        float mean1 = 0;
        float total1 = 0;
        for (int i = 0; i < m_centers.row(1).cols; i++) {
            mean1 += m_centers.at<float>(1, i) * i;
            total1 += m_centers.at<float>(1, i);
        }
        mean1 = mean1 / total1;

        tmd::debug("FeatureComparator", "computeColorCentersIndexes",
                   "mean0 = " + std::to_string(mean0));
        tmd::debug("FeatureComparator", "computeColorCentersIndexes",
                   "mean1 = " + std::to_string(mean1));

        if (mean0 < mean1) {
            m_redCenterIndex = 0;
            m_greenCenterIndex = 1;
        }
        else {
            m_redCenterIndex = 1;
            m_greenCenterIndex = 0;
        }

        tmd::debug("FeatureComparator", "computeColorCentersIndexes",
                   "m_greenCenterIndex = " + std::to_string(m_greenCenterIndex));
        tmd::debug("FeatureComparator", "computeColorCentersIndexes",
                   "m_redCenterIndex = " + std::to_string(m_redCenterIndex));
    }

    void FeatureComparator::detectTeamForPlayers(std::vector<player_t *>
                                                 players) {
        size_t player_count = players.size();
        for (size_t i = 0; i < player_count; i++) {
            detectTeamForPlayer(players[i]);
        }
    }

    void FeatureComparator::detectTeamForPlayer(player_t *player) {
        if (player->features.body_parts.size() == 0) {
            player->team = TEAM_UNKNOWN;
        }
        player->team = getClosestCenter(player) == m_redCenterIndex ?
                       TEAM_A : TEAM_B;
    }
}