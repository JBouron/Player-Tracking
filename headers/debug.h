#ifndef TEAM_MEMBERSHIP_DETECTOR_DEBUG_H
#define TEAM_MEMBERSHIP_DETECTOR_DEBUG_H

#include <string>
#include <iostream>

#define TMD_DEBUG_ENABLE_DEBUG true

namespace tmd{
    void debug(std::string message, bool line_return = true);

    void debug(std::string class_name, std::string method_name,
               std::string message, bool line_return = true);

    /** Function definition here due to inline **/

    inline void debug(std::string message, bool line_return) {
        if (TMD_DEBUG_ENABLE_DEBUG) {
            std::cout << message;
            if (line_return) {
                std::cout << std::endl;
            }
        }
    }

    inline void debug(std::string class_name, std::string method_name,
                      std::string message, bool line_return) {
        if (TMD_DEBUG_ENABLE_DEBUG) {
            std::cout << class_name + "::" + method_name + "() : " + message;
            if (line_return) {
                std::cout << std::endl;
            }
        }
    }
}

#endif //TEAM_MEMBERSHIP_DETECTOR_DEBUG_H
