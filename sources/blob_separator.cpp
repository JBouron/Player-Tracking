#include "../headers/blob_separator.h"

namespace tmd{
    std::vector<tmd::player_t*> BlobSeparator::separate_blobs(
            std::vector<tmd::player_t *> players) {
        std::vector<player_t*> new_player_vector;

        size_t size = players.size();

        // freed
        DPMPlayerExtractor* playerExtractor = new DPMPlayerExtractor();

        for (size_t i = 0 ; i < size ; i ++){
            player_t* p = players[i]; // freed
            cv::imwrite("./res/debug/last_player_image_blob_separator.jpg ",
                        p->original_image);
            if (p->original_image.rows < 100 || p->original_image.cols < 50){
                free_player(p);
                continue;
            }
            frame_t* blob_frame = new frame_t; // freed
            blob_frame->original_frame = p->original_image;
            blob_frame->mask_frame = p->mask_image;
            blob_frame->frame_index = p->frame_index;
            cv::Mat colored_mask =
                    tmd::get_colored_mask_for_frame(blob_frame);
            blob_frame->original_frame.release();
            blob_frame->original_frame = colored_mask;


            tmd::debug("BlobSeparator", "separate_blobs", "Extract players "
                    "from blob.");
            std::vector<player_t*> players_in_blob =
                    playerExtractor->extract_player_from_frame(blob_frame);
            // freed
            tmd::debug("BlobSeparator", "separate_blobs", "Done : " +
                    std::to_string(players_in_blob.size()) + " players "
                                                                 "extracted.");

            if (players_in_blob.size() == 1){
                new_player_vector.push_back(p);
                free_player(players_in_blob[0]);
            }
            else{
                // Here the blob has multiple players in it.
                cv::Mat frame = p->original_image.clone();
                for (size_t j = 0 ; j < players_in_blob.size() ; j ++){
                    tmd::debug("BlobSeparator", "separate_blobs", "Player " +
                            std::to_string(j) + " has score " +
                            std::to_string(players_in_blob[j]->likelihood));

                    player_t* pi = players_in_blob[j];
                    pi->pos_frame.x += p->pos_frame.x;
                    pi->pos_frame.y += p->pos_frame.y;

                    new_player_vector.push_back(pi);
                }
                free_player(p);
                frame.release();
            }
            free_frame(blob_frame);
        }
        delete playerExtractor;
        return new_player_vector;
    }
}