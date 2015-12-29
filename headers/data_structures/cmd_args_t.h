#ifndef BACHELOR_PROJECT_CMD_ARGS_T_H
#define BACHELOR_PROJECT_CMD_ARGS_T_H

#include <string>

namespace tmd{

    /**
     * Structure holding the command line arguments given to the program.
     */
    typedef struct{
        std::string video_folder;
        int camera_index;
        bool show_results = false;
        bool save_results = false;
        bool show_torsos = false;
        int s = 0;
        int e = -1;
        int j = 1;
        int t = 1;
        int b = -1;
    }cmd_args_t;
}

#endif //BACHELOR_PROJECT_CMD_ARGS_T_H