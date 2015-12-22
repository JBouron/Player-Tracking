#include <opencv2/objdetect/objdetect.hpp>
#include "../headers/blob_separator.h"
#include "../headers/pipeline.h"
#include "../headers/debug.h"

namespace tmd{

    int _min(int a, int b){
        return a < b ? a : b;
    }

    int _max(int a, int b){
        return a > b ? a : b;
    }

    std::vector<tmd::player_t*> BlobSeparator::separate_blobs(
            std::vector<tmd::player_t *> players) {
        std::vector<player_t*> new_player_vector;

        size_t size = players.size();
        cv::LatentSvmDetector* detector = new cv::LatentSvmDetector(); // freed
        std::string xml = "./res/xmls/person.xml";
        std::vector<std::string> model_files;
        model_files.push_back(xml);
        detector->load(model_files);

        /*for (size_t i = 0 ; i < size ; i ++){
            cv::imshow("Player", players[i]->original_image);
            cv::waitKey(0);
        }*/

        for (size_t i = 0 ; i < size ; i ++){
            player_t* p = players[i]; // freed
            cv::imwrite("./res/debug/last_player_image_blob_separator.jpg ",
                        p->original_image);
            if (p->original_image.rows < 100 || p->original_image.cols < 50){
                continue;
            }
            frame_t* blob_frame = new frame_t; // freed
            blob_frame->original_frame = p->original_image;
            blob_frame->mask_frame = p->mask_image;
            blob_frame->frame_index = p->frame_index;
            blob_frame->original_frame =
                    tmd::Pipeline::get_colored_mask_for_frame(blob_frame);

            DPMPlayerExtractor* playerExtractor = new DPMPlayerExtractor(
                                                    "./res/xmls/person.xml");

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
                    cv::Rect pos = players_in_blob[j]->pos_frame;
                    pos.x -= 20;
                    pos.y -= 20;
                    pos.width += 40;
                    pos.height += 40;

                    pos.x = _max(0, pos.x);
                    pos.y = _max(0, pos.y);
                    pos.width = _min(frame.cols - pos.x, pos.width);
                    pos.height = _min(frame.rows - pos.y, pos.height);

                    player_t* pi = new player_t;
                    pi->frame_index = p->frame_index;
                    pi->original_image = frame.clone()(pos);
                    pi->mask_image = p->mask_image.clone()(pos);
                    pos.x += p->pos_frame.x;
                    pos.y += p->pos_frame.y;
                    pi->pos_frame = pos;
                    new_player_vector.push_back(pi);
                    free_player(players_in_blob[j]);
                }
                free_player(p);
                frame.release();
            }
            free_frame(blob_frame);
        }
        delete detector;
        return new_player_vector;
    }
}