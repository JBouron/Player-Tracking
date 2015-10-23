#ifndef TEAM_MEMBERSHIP_DETECTOR_BOX_T_H
#define TEAM_MEMBERSHIP_DETECTOR_BOX_T_H

#include "position_t.h"
#include "team_t.h"

namespace tmd{
    /* Structure representing a box.
     * Box are around a player and its color depends on the team of the
     * corresponding player.
     */
    typedef struct{
        position_t pos;     // Position of the box on the frame. Upper left corner.
        position_t size;    // Size of the box in pixels (x = width, y = height).
        team_t color;       /* Color of the box, each team has a different color
                                     to differentiate them. */
    }box_t;
}

#endif //TEAM_MEMBERSHIP_DETECTOR_BOX_T_H
