#include "../../../headers/players_extraction/blob_based_extraction/blob_separator.h"
#include "../../../headers/features_extraction/dpm.h"
#include "../../../headers/data_structures/player_t.h"
#include "../../../headers/data_structures/frame_t.h"

namespace tmd {
    std::vector<tmd::player_t *> BlobSeparator::separate_blobs(
            std::vector<tmd::player_t *> players) {
        std::vector<player_t *> new_player_vector;

        size_t size = players.size();

        // freed
        DPM *dpm = new DPM();

        for (size_t i = 0; i < size; i++) {
            player_t *p = players[i]; // freed
            if (p->original_image.rows < 100 || p->original_image.cols < 50) {
                free_player(p);
                continue;
            }
            frame_t *blob_frame = new frame_t; // freed
            blob_frame->original_frame = p->original_image;
            blob_frame->mask_frame = p->mask_image;
            blob_frame->frame_index = p->frame_index;
            cv::Mat colored_mask =
                    tmd::get_colored_mask_for_frame(blob_frame);
            blob_frame->original_frame.release();
            blob_frame->original_frame = colored_mask;

            tmd::debug("BlobSeparator", "separate_blobs", "Extract players "
                    "from blob.");
            std::vector<player_t *> players_in_blob =
                    dpm->extract_players_and_body_parts(blob_frame);
            // freed
            tmd::debug("BlobSeparator", "separate_blobs", "Done : " +
                                                          std::to_string(players_in_blob.size()) + " players "
                                                                  "extracted.");
            // Here the blob has multiple players in it.
            for (size_t j = 0; j < players_in_blob.size(); j++) {
                tmd::debug("BlobSeparator", "separate_blobs", "Player " +
                                                              std::to_string(j) + " has score " +
                                                              std::to_string(players_in_blob[j]->likelihood));

                player_t *pi = players_in_blob[j];
                pi->pos_frame.x += p->pos_frame.x;
                pi->pos_frame.y += p->pos_frame.y;
                new_player_vector.push_back(pi);
            }
            free_player(p);
            free_frame(blob_frame);
        }
        delete dpm;
        return new_player_vector;
    }
}