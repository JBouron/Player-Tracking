#include "../../headers/test_cases/dpm_detector_tests.h"
#include "../../headers/player_t.h"
#include "../../headers/features_t.h"

namespace tmd{
    void DPMDetectorTest::setUp() {
        m_player_image = cv::imread
                ("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/res/"
                         "tests/player0.jpg");
        m_model_file = "/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project"
                "/res/xmls/person.xml";
        m_player = new player_t;
        m_player->original_image = m_player_image;
        m_dummy_image = cv::imread
                ("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/res/"
                         "tests/dummyimage.jpg");
    }

    void DPMDetectorTest::tearDown() {
        delete m_player;
    }

    void DPMDetectorTest::testInvalidModelFileName() {
        tmd::DPMDetector detector("invalidname");
    }

    void DPMDetectorTest::testRightNumberOfBodyPartExtracted() {
        tmd::DPMDetector detector(m_model_file);
        detector.extractBodyParts(m_player);
        CPPUNIT_ASSERT(m_player->features.body_parts.size() == 6);
    }

    void DPMDetectorTest::testNoFalsePositive() {
        player_t* player = new player_t;
        player->original_image = m_dummy_image;
        tmd::DPMDetector detector(m_model_file);
        detector.extractBodyParts(player);
        CPPUNIT_ASSERT(player->features.body_parts.size() == 0);
    }
}