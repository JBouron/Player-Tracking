#ifndef TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H
#define TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H

namespace tmd{
    /* Enumeration for the different teams present on the field. */
    typedef enum{
        TEAM_A,         // Team A.
        TEAM_B,         // Team B.
        TEAM_REFEREE,   // The referee is counted as a team of one person.
        TEAM_UNKNOWN    // Used when the algorithm couldn't determine the team of the player.
    }team_t;
}

#endif //TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H
