#ifndef BACHELOR_PROJECT_FEATURE_COMPARATOR_H
#define BACHELOR_PROJECT_FEATURE_COMPARATOR_H

#include "../data_structures/features_t.h"
#include "../data_structures/player_t.h"
#include "../features_extraction/features_extractor.h" // Threshold values
#include <opencv2/core/core.hpp>
#include <fstream>
#include <bits/stream_iterator.h>

namespace tmd {

    /**
     * Class responsible for comparing features of player by using a
     * clustering algorithm.
     */
    class FeatureComparator {
    public:

        /**
         * Default constructor for a feature comparator.
         * input :
         *      - data : A matrix where every line is a feature.
         *      - clusterCount : The number of wanted clusters.
         *      - labels : The matrix in which every cluster index is stored
         *      for each sample.
         *      - criteria : A TermCriteria that defines when to stop running
         *      the algorithm.
         *      - attempts : The number of attempts that should be done.
         *      - flags : A flag that defines how the starting centers are
         *      chosen.
         *      - centers : The matrix in which all the centers of the
         *      cluster will be stored once runClustering() is called.
         */
        FeatureComparator(cv::Mat data, int clusterCount, cv::Mat labels,
                          cv::TermCriteria criteria, int attempts,
                          int flags, cv::Mat centers);

        FeatureComparator(int clusterCount, int sampleCols, cv::Mat centers);

        /**
                 * Destructor for the class.
                 */
        ~FeatureComparator();

        /**
         * Runs a kmean clustering algorithm with the parameters given to the
         * comparator at it's construction or later with a set method.
         */
        void runClustering();

        /**
         * Adds the player's features to the data that will be used for the
         * clustering  algorithm.
         */
        void addPlayerFeatures(player_t *player);

        /**
         * Setter method for the termination criteria of the algorithm.
         */
        void setTermCriteria(cv::TermCriteria criteria);

        /**
         * Setter method for the number of attempts the algorithm should do.
         */
        void setAttempts(int attempts);

        /**
         * Setter method for the flag that should be used when running the
         * algorithm.
         */
        void setFlags(int flags);

        /**
         * Getter method for the data that is in the cluster.
         */
        cv::Mat getData();

        /**
         * Writes the cluster's centers to a file "clusterCenters<frame_index>.txt".
         */
        void writeCentersToFile(int frame_index);

        /**
         * Writes the cluster's centers to a file "clusterCenters<frame_index>.txt".
         */
        void writeCentersToFile();

        /**
         * Reads the cluster's centers from a file "clusterCenters.txt".
         */
        static cv::Mat readCentersFromFile();

        /**
         * Detects the teams of a collection of players.
         */
        void detectTeamForPlayers(std::vector<player_t *> players);

        /**
         * Detects the team of the given player.
         */
        void detectTeamForPlayer(player_t *player);

    private:
        int m_sampleCols;
        cv::Mat m_data;
        int m_clusterCount;
        cv::Mat m_labels;
        cv::TermCriteria m_termCriteria;
        int m_attempts;
        int m_flags;
        cv::Mat m_centers;
        int m_redCenterIndex;
        int m_greenCenterIndex;

        /**
         * Returns the Mat corresponding to a player's features.
         */
        cv::Mat getMatForPlayerFeature(player_t *player);

        /**
         * Returns a vector<float> of all of the floats contained in a string.
         */
        static std::vector<float> getFloatsFromString(std::string inputString);

        /**
         * Make the mapping between colors and centers.
         */
        void computeColorCentersIndexes();

        /**
         * Adds a sample to the data will be used for the clustering algorithm.
         */
        void addSampleToData(cv::Mat sample);

        /**
         * Gets the closest center of the clusters for a sample.
         * Returns the index of the closest center in m_centers.
         */
        int getClosestCenter(cv::Mat sample);

        /**
         * Gets the closest center of the clusters for a player.
         * Returns the index of the closest center in m_centers.
         */
        int getClosestCenter(player_t *player);

    };
}

#endif //BACHELOR_PROJECT_FEATURE_COMPARATOR_H
