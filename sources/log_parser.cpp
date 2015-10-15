#include <stdexcept>
#include <string.h>
#include <sstream>
#include "../headers/log_parser.h"
#include "../headers/debug.h"

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

    std::vector<int> LogParser::get_next_frame_positions() {
        int current_log_frame = m_current_frame;
        m_current_frame ++;
        std::string line;
        char frame_idx[10];
        char grid_index[10];
        char player_pos_x[10];
        char player_pos_y[10];
        std::vector<int> pos;
        while (current_log_frame != m_current_frame){
            if (m_log.eof()){
                break;
            }
            std::streampos last_tellg = m_log.tellg();
            getline(m_log, line);
            tmd::debug("LogParser", "get_next_frame_positions", "current_log_frame = " + std::to_string(current_log_frame) + "  line = " + line);
            char* ptr = strtok((char*) line.c_str(), " ");
            if (ptr != NULL) strcpy(frame_idx, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");
            if (ptr != NULL) strcpy(grid_index, ptr); else throw std::runtime_error("Error, invalid log format : " + line);
            ptr = strtok(NULL, " ");

            tmd::debug("LogParser", "get_next_frame_positions", "frame_idx = " + std::string(frame_idx) + "    grid_index = " + std::string(grid_index));
            std::stringstream strs;
            strs << frame_idx;
            strs >> current_log_frame;
            tmd::debug("LogParser", "get_next_frame_positions", "frame idx = " + std::to_string(current_log_frame));
            strs.clear();
            strs << grid_index;
            int cell;
            strs >> cell;
            tmd::debug("LogParser", "get_next_frame_positions", "cell = " + std::to_string(cell));
            if (current_log_frame != m_current_frame){
                pos.push_back(cell);
            }
            else{
                m_log.seekg(last_tellg);
            }
        }
        return pos;
    }

    unsigned int LogParser::get_current_frame_index() {
        return m_current_frame - 1;
    }
}