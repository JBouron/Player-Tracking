#ifndef BACHELOR_PROJECT_TEST_SUITE_H
#define BACHELOR_PROJECT_TEST_SUITE_H

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include "bgsubstractor_tests.h"

namespace tmd{
    /** test suite of the project
     * Allows us to run all the tests from qone place.
     * */

    void run_tests(void);

    void run_tests_bgs(void);
    void run_tests_dpm(void);
}

#endif //BACHELOR_PROJECT_TEST_SUITE_H
