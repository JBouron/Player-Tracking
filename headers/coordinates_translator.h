#ifndef TEAM_MEMBERSHIP_DETECTOR_COORDINATES_TRANSLATOR_H
#define TEAM_MEMBERSHIP_DETECTOR_COORDINATES_TRANSLATOR_H

#include "position_t.h"

namespace tmd{
    /** This class is responsible to convert 3D positions (from the camera image)
     * to 2D positions (field cells).
     */

    class CoordinatesTranslator{
    public:
        static position_t convert(position_t* 3d_pos, unsigned char camera_id);

    private:
        // TODO : Add js interpreter.
    };
}

#endif //TEAM_MEMBERSHIP_DETECTOR_COORDINATES_TRANSLATOR_H
