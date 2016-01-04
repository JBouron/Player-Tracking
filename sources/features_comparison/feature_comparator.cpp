#include "../../headers/features_comparison/feature_comparator.h"

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
                                          10, 1.0);
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

    int FeatureComparator::getClosestCenter(cv::Mat sample) {
        if (m_data.cols != sample.cols || sample.rows != 1) {
            throw std::invalid_argument(
                    "Sample doesn't have the same amount of dimensions as the data !");
        }

        double distances[m_centers.rows];
        double max = 0;
        int index_max = -1;
        for (int i = 0; i < m_centers.rows; i++) {
            distances[i] = compareHist(m_centers.row(i), sample, CV_COMP_CORREL);
            tmd::debug("FeatureComparator", "getClosestCenter", "Correlation "
                                                                        "with center = " +
                                                                std::to_string(distances[i]));
            tmd::debug("FeatureComparator", "getClosestCenter", "Center "
                                                                        "index = " + std::to_string(i));
            if (distances[i] > max) {
                max = distances[i];
                index_max = i;
            }
        }
        tmd::debug("FeatureComparator", "getClosestCenter", "max = " +
                                                            std::to_string(max));

        if (max < Config::features_comparator_correlation_threshold) {
            index_max = -1;
        }
        return index_max;
    }

    int FeatureComparator::getClosestCenter(player_t *player) {
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
        std::ofstream clustersFile(tmd::Config::features_comparator_centers_file_name);
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
            clustersFile.flush();
            clustersFile.close();
        }
    }

    void FeatureComparator::writeCentersToFile(int frame_index) {
        std::ofstream clustersFile(tmd::Config::features_comparator_centers_file_name + std::to_string(frame_index));
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
            clustersFile.flush();
            clustersFile.close();
        }
    }

    Mat FeatureComparator::readCentersFromFile() {
        int rows = tmd::Config::features_comparator_centers_file_rows;
        int cols = tmd::Config::features_comparator_centers_file_cols;
        std::ifstream clustersFile(tmd::Config::features_comparator_centers_file_name);
        if (!clustersFile.is_open()) {
            throw std::runtime_error("Error couldn't load " + tmd::Config::features_comparator_centers_file_name);
        }
        Mat toReturn(rows, cols, CV_32F);
        if (clustersFile.is_open()) {
            string line;
            for (int j = 0; j < rows; j++) {
                getline(clustersFile, line);
                vector<float> floatVector = getFloatsFromString(line);
                for (int i = 0; i < cols; i++) {
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
        else {
            int team_index = getClosestCenter(player);

            if (team_index == -1) {
                player->team = TEAM_UNKNOWN;
            }
            else if (team_index == m_redCenterIndex) {
                player->team = TEAM_A;
            }
            else if (team_index == m_greenCenterIndex) {
                player->team = TEAM_B;
            }
            else {
                player->team = TEAM_UNKNOWN;
            }
        }
    }
}