#include <stdexcept>
#include <string.h>
#include "../headers/log_parser.h"

namespace tmd{
    LogParser::LogParser(std::string log_path) {
        m_log.open(log_path);
        if (!m_log.is_open()){
            throw std::invalid_argument("Error couldn't load log file + \"" + log_path + "\"");
        }
        m_current_frame = 0;
        std::string first_line;
        getline(m_log, first_line);
        if (!(first_line == "frame_number grid_index player_position_x_meters player_position_y_meters")){
            throw std::invalid_argument("Error invalid log file format. First line is : " + first_line);
        }
    }

    LogParser::~LogParser() {
        m_log.close();
    }

    std::vector<tmd::position_t> LogParser::get_next_frame_positions() {
        m_current_frame ++;
        std::string line;
        int current_log_frame = m_current_frame;
        char frame_idx[10];
        char grid_index[10];
        char player_pos_x[10];
        char player_pos_y[10];
        std::vector<position_t> pos;
        while (current_log_frame != m_current_frame + 1){
            if (m_log.eof()){
                break;
            }

            getline(m_log, line);
            char* ptr = strtok((char *) line.c_str(), " ");
            if (ptr != NULL) strcpy(frame_idx, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");
            if (ptr != NULL) strcpy(grid_index, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");
            if (ptr != NULL) strcpy(player_pos_x, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");
            if (ptr != NULL) strcpy(player_pos_y, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");

            
        }
    }
}