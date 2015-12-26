#ifndef BACHELOR_PROJECT_CONFIG_H
#define BACHELOR_PROJECT_CONFIG_H

#include <libconfig.h++>

#define TMD_CONFIG_FILE "./res/config.cfg"

namespace tmd{
    /**
     * Class representing the config of the program.
     */
    class Config{
    public:
        /**
         * Load the config from the config file.
         */
        static void load_config();

        /**
         * Here are all configs / parameters / values ...
         * They are public for ease of use.
         */

        static bool bgs_detect_shadows;
    };
}

#endif //BACHELOR_PROJECT_CONFIG_H
