#include <opencv2/objdetect/objdetect.hpp>
#include "../headers/blob_separator.h"
#include "../headers/player_t.h"
#include "../headers/dpm_player_extractor.h"

namespace tmd{

    int min(int a, int b){
        return a < b ? a : b;
    }

    int max(int a, int b){
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
            std::vector<cv::LatentSvmDetector::ObjectDetection> results;
            detector->detect(p->original_image, results,
                         TMD_DEFAULT_DMP_EXTRACTOR_OVERLAPPING_THRESHOLD, 4);

            std::vector<cv::LatentSvmDetector::ObjectDetection>
                    filtered_results;
            for (size_t j = 0 ; j < results.size() ; j ++){
                if (results[i].score >=
                    TMD_DEFAULT_DMP_EXTRACTOR_SCORE_THRESHOLD){
                    filtered_results.push_back(results[i]);
                }
            }

            if (filtered_results.size() == 1){
                new_player_vector.push_back(p);
            }
            else{
                // Here the blob has multiple players in it.
                cv::Mat frame = p->original_image;
                for (size_t j = 0 ; j < filtered_results.size() ; j ++){
                    cv::Rect pos = filtered_results[i].rect;
                    pos.x -= 20;
                    pos.y -= 20;
                    pos.width += 40;
                    pos.height += 40;

                    pos.x = max(0, pos.x);
                    pos.y = max(0, pos.y);
                    pos.width = min(frame.cols - pos.x, pos.width);
                    pos.height = min(frame.rows - pos.y, pos.height);

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