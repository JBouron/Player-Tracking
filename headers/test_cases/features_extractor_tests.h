#ifndef BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H
#define BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include "../features_extractor.h"

namespace tmd{
    /** Test Class for the FeaturesExtractor **/
    class FeaturesExtractorTest : public CppUnit::TestFixture{
        CPPUNIT_TEST_SUITE(FeaturesExtractorTest);
        CPPUNIT_TEST(testExtractBodyPartsCorrectNumberOfParts);
        CPPUNIT_TEST(testHSVConversionIsCorrectForHue);
        CPPUNIT_TEST(testHSVConversionIsCorrectForSaturation);
        CPPUNIT_TEST(testHSVConversionIsCorrectForValue);
        CPPUNIT_TEST(testMaskIsUpdatedInRespectWithThresholds);
        CPPUNIT_TEST(testHistogramsContainsSameNumberOfValues);
        CPPUNIT_TEST_SUITE_END();
    public:
        void setUp();
        void tearDown();

    protected:
        void testExtractBodyPartsCorrectNumberOfParts();
        void testHSVConversionIsCorrectForHue();
        void testHSVConversionIsCorrectForSaturation();
        void testHSVConversionIsCorrectForValue();
        void testMaskIsUpdatedInRespectWithThresholds();
        void testHistogramsContainsSameNumberOfValues();

    private:
        tmd::player_t* m_player;
        cv::Mat m_player_image;
        // Image with hue= 120, sat. = val. = 1.0.
        cv::Mat m_hue_120_image;
        cv::Mat m_sat_05_image;
        cv::Mat m_val_05_image;
        std::string m_model_file;
    };
}

#endif //BACHELOR_PROJECT_FEATURES_EXTRACTOR_TESTS_H
