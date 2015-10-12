#ifndef TEAM_MEMBERSHIP_DETECTOR_POSITION_T_H
#define TEAM_MEMBERSHIP_DETECTOR_POSITION_T_H

namespace tmd{
    /* Structure holding coordinates of a position.
     * The position can be in pixels or in the field.
     */
    typedef struct{
        int x;
        int y;
    }position_t;
}

#endif //TEAM_MEMBERSHIP_DETECTOR_POSITION_T_H
