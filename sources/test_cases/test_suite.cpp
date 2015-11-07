#include "../../headers/test_cases/test_suite.h"

namespace tmd {

    void run_tests(void) {
        CPPUNIT_TEST_SUITE_REGISTRATION(BGSubstractorTest);
        CppUnit::TextUi::TestRunner runner;
        CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
        runner.addTest(registry.makeTest());
        bool wasSuccessful = runner.run("", false);
    }
}
