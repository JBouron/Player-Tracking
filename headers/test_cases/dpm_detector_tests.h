#include <cppunit/TestFixture.h>
#include <cppunit/Test.h>
#include <cppunit/extensions/HelperMacros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../players_extraction/dpm_based_extraction/dpm_detector.h"

#ifndef BACHELOR_PROJECT_DPM_DETECTOR_TESTS_H
#define BACHELOR_PROJECT_DPM_DETECTOR_TESTS_H

namespace tmd{
    /** Test class for the DPMDetector. **/
    class DPMDetectorTest : public CppUnit::TestFixture{
        CPPUNIT_TEST_SUITE(DPMDetectorTest);
        CPPUNIT_TEST(testRightNumberOfBodyPartExtracted);
        CPPUNIT_TEST(testNoFalsePositive);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

    protected:
        void testRightNumberOfBodyPartExtracted();
        void testNoFalsePositive();

    private:
        // Image that we know there is a player on it and the detector should
        // be able to detect him and its body parts.
        cv::Mat m_player_image;
        tmd::player_t* m_player;
        std::string m_model_file;
        // Dummy image not containing any player.
        cv::Mat m_dummy_image;
    };
}

#endif //BACHELOR_PROJECT_DPM_DETECTOR_TESTS_H
