#include <opencv2/objdetect/objdetect.hpp>
#include "../headers/blob_separator.h"
#include "../headers/player_t.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/frame_t.h"
#include "../headers/pipeline.h"

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
        cv::LatentSvmDetector* detector = new cv::LatentSvmDetector();
        std::string xml = "./res/xmls/person.xml";
        std::vector<std::string> model_files;
        model_files.push_back(xml);
        detector->load(model_files);

        for (size_t i = 0 ; i < size ; i ++){
            player_t* p = players[i];

            frame_t* blob_frame = new frame_t;
            blob_frame->original_frame = p->original_image;
            blob_frame->mask_frame = p->mask_image;
            blob_frame->frame_index = p->frame_index;
            blob_frame->original_frame =
                    tmd::Pipeline::get_colored_mask_for_frame(blob_frame);

            cv::imshow("mask", blob_frame->mask_frame);
            cv::waitKey(0);
            cv::imshow("frame", blob_frame->original_frame);
            cv::waitKey(0);

            DPMPlayerExtractor* playerExtractor = new DPMPlayerExtractor(
                                                    "./res/xmls/person.xml");

            std::vector<player_t*> players_in_blob =
                    playerExtractor->extract_player_from_frame(blob_frame);

            if (players_in_blob.size() == 1){
                new_player_vector.push_back(p);
            }
            else{
                // Here the blob has multiple players in it.
                cv::Mat frame = p->original_image;
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
                    pi->pos_frame = pos;
                    pi->frame_index = p->frame_index;
                    pi->original_image = frame.clone()(pos);
                    pi->mask_image = p->mask_image.clone()(pos);
                    new_player_vector.push_back(pi);
                }
            }
        }
        return new_player_vector;
    }
}