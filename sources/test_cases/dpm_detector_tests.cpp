#include "../../headers/test_cases/dpm_detector_tests.h"
#include "../../headers/player_t.h"
#include "../../headers/features_t.h"

namespace tmd{
    void DPMDetectorTest::setUp() {
        m_player_image = cv::imread("./res/tests/player0.jpg");
        m_model_file = "./res/xmls/person.xml";
        m_player = new player_t;
        m_player->original_image = m_player_image;
        const int rows = m_player->original_image.rows;
        const int cols = m_player->original_image.cols;
        m_player->mask_image = cv::Mat(rows, cols, CV_8U);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m_player->mask_image.at<uchar>(i, j) = 255;
            }
        }
        m_dummy_image = cv::imread("./res/tests/dummyimage.jpg");
    }

    void DPMDetectorTest::tearDown() {
        delete m_player;
    }

    void DPMDetectorTest::testRightNumberOfBodyPartExtracted() {
        tmd::DPMDetector detector;
        detector.extractBodyParts(m_player);
        CPPUNIT_ASSERT(m_player->features.body_parts.size() == 6);
    }

    void DPMDetectorTest::testNoFalsePositive() {
        player_t* player = new player_t;
        player->original_image = m_dummy_image;
        tmd::DPMDetector detector;
        detector.extractBodyParts(player);
        CPPUNIT_ASSERT(player->features.body_parts.size() == 0);
    }
}