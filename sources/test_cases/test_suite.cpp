#include "../../headers/test_cases/test_suite.h"
#include "../../headers/test_cases/dpm_detector_tests.h"
#include "../../headers/debug.h"
#include "../../headers/test_cases/features_extractor_tests.h"

namespace tmd {

    void run_tests(void) {
        run_tests_bgs();
        run_tests_dpm();
        run_tests_features_extractor();
    }

    void run_tests_bgs(void){
        CPPUNIT_TEST_SUITE_REGISTRATION(BGSubstractorTest);
        tmd::debug("Running tests for BGSubstractor.");
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry =
                CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        runner.run("", false);
        tmd::debug("Tests for BGSubstractor completed.");
    }

    void run_tests_dpm(void){
        tmd::debug("Running tests for DPMDetector.");
        CPPUNIT_TEST_SUITE_REGISTRATION(DPMDetectorTest);
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry =
                CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        runner.run("", false);
        tmd::debug("Tests for DPMDetector completed.");
    }

    void run_tests_features_extractor(void){
        tmd::debug("Running tests for FeaturesExtractor.");
        CPPUNIT_TEST_SUITE_REGISTRATION(FeaturesExtractorTest);
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry =
                CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        runner.run("", false);
        tmd::debug("Tests for FeaturesExtractor completed.");
    }
}
