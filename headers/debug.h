#ifndef TEAM_MEMBERSHIP_DETECTOR_DEBUG_H
#define TEAM_MEMBERSHIP_DETECTOR_DEBUG_H

#include <string>
#include <iostream>

#define TMD_DEBUG_ENABLE_DEBUG true

namespace tmd{
    void debug(std::string message, bool line_return = true);

    void debug(std::string class_name, std::string method_name,
               std::string message, bool line_return = true);
}

#endif //TEAM_MEMBERSHIP_DETECTOR_DEBUG_H
