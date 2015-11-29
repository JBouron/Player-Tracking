#include "../../headers/test_cases/test_suite.h"
#include "../../headers/test_cases/dpm_detector_tests.h"
#include "../../headers/debug.h"
#include "../../headers/test_cases/features_extractor_tests.h"
#include "../../headers/test_cases/feature_comparator_tests.h"

namespace tmd {

    void run_tests(void) {
        // TODO : Resolve tests changes and uncomment this.
//        run_tests_feature_comparator();
        run_tests_bgs();
        run_tests_dpm();
        run_tests_features_extractor();
    }

    void run_tests_bgs(void){
        tmd::debug("Running tests for BGSubstractor.");
        CPPUNIT_TEST_SUITE_REGISTRATION(BGSubstractorTest);
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry =
                CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        runner.run("", false);
        tmd::debug("Tests for BGSubstractor completed.");
    }

    // TODO : Resolve tests and uncomment this.
    /*void run_tests_feature_comparator(void){
        tmd::debug("Running tests for FeatureComparator.");
        CPPUNIT_TEST_SUITE_REGISTRATION(FeatureComparatorTest);
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry =
                CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        runner.run("", false);
        tmd::debug("Tests for FeatureComparator completed.");
    }*/

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
