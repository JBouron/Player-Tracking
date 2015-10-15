#ifndef BACHELOR_PROJECT_LOG_PARSER_H
#define BACHELOR_PROJECT_LOG_PARSER_H

#include <string>
#include <vector>
#include <fstream>
#include "position_t.h"

namespace tmd{
    /** Class parsing the log containing the player position. */
    class LogParser{
    public:
        LogParser(std::string log_path);
        ~LogParser();

        std::vector<int> get_next_frame_positions();

        unsigned int get_current_frame_index();

    private:
        std::ifstream m_log;
        unsigned int m_current_frame;
    };
}

#endif //BACHELOR_PROJECT_LOG_PARSER_H
