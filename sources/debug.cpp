#include "../headers/debug.h"

namespace tmd {

    void debug(std::string message, bool line_return) {
        if (TMD_DEBUG_ENABLE_DEBUG) {
            std::cout << message;
            if (line_return) {
                std::cout << std::endl;
            }
        }
    }

    void debug(std::string class_name, std::string method_name,
               std::string message, bool line_return) {
        if (TMD_DEBUG_ENABLE_DEBUG) {
            std::cout << class_name + "::" + method_name + "() : " + message;
            if (line_return) {
                std::cout << std::endl;
            }
        }
    }
}