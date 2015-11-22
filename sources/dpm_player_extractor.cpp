#include <iostream>
#include "../headers/dpm_player_extractor.h"
#include "../headers/frame_t.h"
#include "../headers/player_t.h"

namespace tmd{
    DPMPlayerExtractor::DPMPlayerExtractor(std::string model_file, float
    overlap_threshold){
        std::vector<std::string> model_files;
        model_files.push_back(model_file);
        m_detector = new cv::LatentSvmDetector();
        bool model_loaded = m_detector->load(model_files);
        if (!model_loaded){
            throw std::invalid_argument("Error in DPMPlayerExtractor "
                                    "constructor, couldn't load model file.");
        }
        m_overlap_threshold = overlap_threshold;
    }

    DPMPlayerExtractor::~DPMPlayerExtractor(){
        m_detector->clear();
        delete m_detector;
    }

    std::vector<player_t*> DPMPlayerExtractor::extract_player_from_frame(
            frame_t *frame){
        cv::Mat image = frame->original_frame;
        std::vector<cv::LatentSvmDetector::ObjectDetection> results;
        m_detector->detect(image, results, m_overlap_threshold, 4);

        std::vector<tmd::player_t*> players;

        for (size_t i = 0 ; i < results.size() ; i ++){
            if (results[i].score > TMD_DMP_EXTRACTOR_SCORE_THRESHOLD) {
                players.push_back(new player_t);
                tmd::player_t *p = players[i];
                std::cout << results[i].score << std::endl;
                p->frame_index = static_cast<int> (frame->frame_index);

                cv::Rect playerRect = results[i].rect;
                p->original_image = frame->original_frame(playerRect);
                //p->mask_image = frame->mask_frame(playerRect);
            }
        }

        return players;
    }
}