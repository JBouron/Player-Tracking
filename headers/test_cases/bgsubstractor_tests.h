#include <cppunit/TestFixture.h>
#include <cppunit/Test.h>
#include <cppunit/extensions/HelperMacros.h>
#include "../bgsubstractor.h"

#ifndef BACHELOR_PROJECT_BGSUBSTRACTOR_TESTS_H
#define BACHELOR_PROJECT_BGSUBSTRACTOR_TESTS_H

namespace tmd{
    /** Test class for the BGSubstractor **/
    class BGSubstractorTest : public CppUnit::TestFixture{
        CPPUNIT_TEST_SUITE(BGSubstractorTest);
        CPPUNIT_TEST_EXCEPTION(testInvalidVideoInput, std::invalid_argument);
        CPPUNIT_TEST_EXCEPTION(testInvalidCameraIndex1, std::invalid_argument);
        CPPUNIT_TEST_EXCEPTION(testInvalidCameraIndex2, std::invalid_argument);
        CPPUNIT_TEST(testHasNextFrameIsCorrect);
        CPPUNIT_TEST_EXCEPTION(testNextFrameThrowsExceptionWhenNoFramesLeft,
                               std::runtime_error);
        CPPUNIT_TEST(testNextFrameConstructCorrectFramesCameraIndex);
        CPPUNIT_TEST(testNextFrameConstructCorrectFramesFrameIndex);
        CPPUNIT_TEST_SUITE_END();
    public:
        /**
         * setUp is called right before ldaunching the tests.
         * It basically initialize the variables used in the tests.
         */
        void setUp();

        /**
         * tearDown is called after running all the tests.
         * It basically free all initialized variables.
         */
        void tearDown();

    protected:
        /** Here are our tests case for this class **/
        void testInvalidVideoInput();
        void testInvalidCameraIndex1();
        void testInvalidCameraIndex2();
        void testHasNextFrameIsCorrect();
        void testNextFrameThrowsExceptionWhenNoFramesLeft();
        void testNextFrameConstructCorrectFramesCameraIndex();
        void testNextFrameConstructCorrectFramesFrameIndex();

    private:
        cv::VideoCapture* m_video;
    };
}
#endif //BACHELOR_PROJECT_BGSUBSTRACTOR_TESTS_H
