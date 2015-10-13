#ifndef BACHELOR_PROJECT_COORDINATES_TRANSLATOR_H
#define BACHELOR_PROJECT_COORDINATES_TRANSLATOR_H

#include "position_t.h"

namespace tmd{
    /* Class handling transformation from world coordinates to field coordinates
     * and the other way around.
     */

    class CoordinatesTranslator{
    public:
        static position_t world_to_field(position_t w_pos, unsigned char camera_id);
        static position_t field_to_world(position_t f_pos, unsigned char camera_id);
    };
}

#endif //BACHELOR_PROJECT_COORDINATES_TRANSLATOR_H
