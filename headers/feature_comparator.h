#ifndef BACHELOR_PROJECT_FEATURE_COMPARATOR_H
#define BACHELOR_PROJECT_FEATURE_COMPARATOR_H

#include "features_t.h"
#include "player_t.h"

namespace tmd{
    /* Class responsible of comparing features. */

    class FeatureComparator{
    public:
        FeatureComparator(cv::Mat data, int clusterCount, cv::Mat labels, cv::TermCriteria criteria, int attempts,
                          int flags, cv::Mat centers);
        ~FeatureComparator();
        void runClustering();
        void addSampleToData(cv::Mat sample);
        cv::Mat getClosestCenter(cv::Mat sample);
        cv::Mat getClosestCenter(player_t *player, int i);
        void addPlayerFeatures(player_t *player, int i);
        void setTermCriteria(cv::TermCriteria criteria);
        void setAttempts(int attempts);
        void setFlags(int flags);
        void writeCentersToFile();
        cv::Mat readCentersFromFile();

    private:
        cv::Mat getMatForPlayerFeature(player_t *player, int i);
        std::vector<float> getFloatsFromString(std::string inputString);

        cv::Mat m_data;
        int m_clusterCount;
        cv::Mat m_labels;
        cv::TermCriteria m_termCriteria;
        int m_attempts;
        int m_flags;
        cv::Mat m_centers;
   };
}

#endif //BACHELOR_PROJECT_FEATURE_COMPARATOR_H
