#include <opencv2/highgui/highgui.hpp>
#include "../../headers/test_cases/features_extractor_tests.h"
#include "../../headers/test_cases/feature_comparator_tests.h"

namespace tmd{
    void FeatureComparatorTest::setUp() {
        m_data, m_labels = cv::Mat(1,3, CV_32F);
        cv::Mat sample(1, 3, CV_32F);
        sample.at<float>(0,0) = 10;
        sample.at<float>(0,1) = 11;
        sample.at<float>(0,2) = 12;

        m_data.push_back(sample);

        m_clusterCount = 3;
        m_centers = cv::Mat(m_clusterCount, 3, CV_32F);
        m_termCriteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                                       10, 1.0);
        m_attempts = 3;
        m_flags = cv::KMEANS_PP_CENTERS;
    }

    void FeatureComparatorTest::tearDown() {
    }



    void FeatureComparatorTest::testCreateCluster() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);
    }

    void FeatureComparatorTest::testAddSampleToData() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);
        cv::Mat sample(1, 3, CV_32F);
        sample.at<float>(0,0) = 20;
        sample.at<float>(0,1) = 21;
        sample.at<float>(0,2) = 22;

        comparator.addSampleToData(sample);
        CPPUNIT_ASSERT(comparator.getData().rows == 2);
    }

    void FeatureComparatorTest::testAddSampleDataIllegalSample() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);
        cv::Mat sample(1, 2, CV_32F);
        sample.at<float>(0,0) = 20;
        sample.at<float>(0,1) = 21;

        comparator.addSampleToData(sample);
    }

    void FeatureComparatorTest::testGetClosestCenter() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);

        cv::Mat sample(1, 3, CV_32F);
        sample.at<float>(0,0) = 20;
        sample.at<float>(0,1) = 21;
        sample.at<float>(0,2) = 22;

        cv::Mat sample2(1, 3, CV_32F);
        sample2.at<float>(0,0) = 30;
        sample2.at<float>(0,1) = 31;
        sample2.at<float>(0,2) = 32;

        comparator.addSampleToData(sample);
        comparator.addSampleToData(sample2);
        comparator.runClustering();

        cv::Mat sample3(1, 3, CV_32F);
        sample3.at<float>(0,0) = 32;
        sample3.at<float>(0,1) = 30;
        sample3.at<float>(0,2) = 31;

        cv::Mat center = comparator.getClosestCenter(sample3);
        CPPUNIT_ASSERT(center.rows == 1);
        CPPUNIT_ASSERT(center.cols == 3);
        CPPUNIT_ASSERT(center.at<float>(0,0) == 30);
        CPPUNIT_ASSERT(center.at<float>(0,1) == 31);
        CPPUNIT_ASSERT(center.at<float>(0,2) == 32);
    }

    void FeatureComparatorTest::testGetData() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);

        cv::Mat sample(1, 3, CV_32F);
        sample.at<float>(0,0) = 20;
        sample.at<float>(0,1) = 21;
        sample.at<float>(0,2) = 22;

        cv::Mat sample2(1, 3, CV_32F);
        sample2.at<float>(0,0) = 30;
        sample2.at<float>(0,1) = 31;
        sample2.at<float>(0,2) = 32;

        comparator.addSampleToData(sample);
        comparator.addSampleToData(sample2);

        cv::Mat data = comparator.getData();
        CPPUNIT_ASSERT(data.rows == 3);
        CPPUNIT_ASSERT(data.cols == 3);
        CPPUNIT_ASSERT(data.at<float>(0,0) == 10);
        CPPUNIT_ASSERT(data.at<float>(0,1) == 11);
        CPPUNIT_ASSERT(data.at<float>(0,2) == 12);
        CPPUNIT_ASSERT(data.at<float>(1,0) == 20);
        CPPUNIT_ASSERT(data.at<float>(1,1) == 21);
        CPPUNIT_ASSERT(data.at<float>(1,2) == 22);
        CPPUNIT_ASSERT(data.at<float>(2,0) == 30);
        CPPUNIT_ASSERT(data.at<float>(2,1) == 31);
        CPPUNIT_ASSERT(data.at<float>(2,2) == 32);
    }

    void FeatureComparatorTest::testWriteAndReadCenters() {
        FeatureComparator comparator(m_data, m_clusterCount, m_labels, m_termCriteria, m_attempts,
                                     m_flags, m_centers);

        cv::Mat sample(1, 3, CV_32F);
        sample.at<float>(0,0) = 20;
        sample.at<float>(0,1) = 21;
        sample.at<float>(0,2) = 22;

        cv::Mat sample2(1, 3, CV_32F);
        sample2.at<float>(0,0) = 30;
        sample2.at<float>(0,1) = 31;
        sample2.at<float>(0,2) = 32;

        comparator.addSampleToData(sample);
        comparator.addSampleToData(sample2);
        comparator.runClustering();


        comparator.writeCentersToFile();
        cv::Mat centers = comparator.readCentersFromFile(3,3);

        std::cout << centers << std::endl;
        std::cout << centers.rows << std::endl;
        std::cout << centers.cols << std::endl;
        std::cout << centers.type() << std::endl;

        for(int i = 0; i < 3; i ++){
            for(int j = 0; j < 3; j++){
                CPPUNIT_ASSERT(centers.at<float>(i,j) == m_centers.at<float>
                        (i,j));
            }
        }
    }
}