#ifndef TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H
#define TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H

#include <string>

namespace tmd{
    /* Enumeration for the different teams present on the field. */
    typedef enum{
        TEAM_A,         // Team A.
        TEAM_B,         // Team B.
        TEAM_REFEREE,   // The referee is counted as a team of one person.
        TEAM_UNKNOWN    // Used when the algorithm couldn't determine the team of the player.
    }team_t;

    inline std::string get_team_string(team_t team){
        switch (team){
            case TEAM_A         : return "Team A";
            case TEAM_B         : return "Team B";
            case TEAM_REFEREE   : return "Team REFEREE";
            case TEAM_UNKNOWN   : return "Team UNKNOWN";
        }
    }

    inline CvScalar get_team_color(team_t team){
        CvScalar color;
        switch (team){
            case TEAM_A :
                color.val[0] = 0;
                color.val[1] = 0;
                color.val[3] = 255;
                color.val[2] = 255;
                break;

            case TEAM_B :
                color.val[0] = 0;
                color.val[1] = 255;
                color.val[2] = 0;
                color.val[3] = 255;
                break;

            case TEAM_UNKNOWN:
                color.val[0] = 1;
                color.val[1] = 1;
                color.val[2] = 1;
                color.val[3] = 255;
                break;

            default:
                color.val[0] = 0;
                color.val[1] = 0;
                color.val[2] = 0;
                color.val[3] = 255;
                break;
        }
        return color;
    }
}

#endif //TEAM_MEMBERSHIP_DETECTOR_TEAM_T_H
