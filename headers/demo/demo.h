#ifndef BACHELOR_PROJECT_DEMO_H
#define BACHELOR_PROJECT_DEMO_H

#include "../player_t.h"

namespace tmd{
    void run_demo_dpm(void);
    void run_demo_pipeline(void);

    void show_original_image(const tmd::player_t* const player);
    void show_original_image_and_mask(const tmd::player_t* const player);
    void show_dpm_detection_parts(const tmd::player_t* const player);
    void show_torso_part(const tmd::player_t* const player);
    void show_torso_mask_before_th(const tmd::player_t* const player);
    void show_torso_mask_after_th(const tmd::player_t* const player);
    void show_torso_histogram(const tmd::player_t* const player);
}

#endif //BACHELOR_PROJECT_DEMO_H
