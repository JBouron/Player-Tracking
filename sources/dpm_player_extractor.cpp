#include <iostream>
#include "../headers/dpm_player_extractor.h"
#include "../headers/frame_t.h"
#include "../headers/player_t.h"

namespace tmd{
    DPMPlayerExtractor::DPMPlayerExtractor(std::string model_file, float
    overlap_threshold, float score_threshold){
        std::vector<std::string> model_files;
        model_files.push_back(model_file);
        m_detector = new cv::LatentSvmDetector();
        bool model_loaded = m_detector->load(model_files);
        if (!model_loaded){
            throw std::invalid_argument("Error in DPMPlayerExtractor "
                                    "constructor, couldn't load model file.");
        }
        m_overlap_threshold = overlap_threshold;
        if (m_overlap_threshold < 0.0) m_overlap_threshold = 0.0;
        else if (1.0 < m_overlap_threshold) m_overlap_threshold = 1.0;

        m_score_threshold = score_threshold;
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
            if (results[i].score > m_score_threshold) {
                players.push_back(new player_t);
                tmd::player_t *p = players[players.size()-1];
                p->likelihood = results[i].score;
                std::cout << results[i].score << std::endl;
                p->frame_index = static_cast<int> (frame->frame_index);

                cv::Rect playerRect = results[i].rect;
                playerRect.height += 40;
                playerRect.width += 40;
                playerRect.x -= 20;
                playerRect.y -= 20;
                if (playerRect.x < 0) playerRect.x = 0;
                if (playerRect.y < 0) playerRect.y = 0;
                if (playerRect.x + playerRect.width > frame->original_frame
                                                              .cols){
                    playerRect.width = frame->original_frame.cols -
                            playerRect.x;
                }
                if (playerRect.y + playerRect.height > frame->original_frame
                        .rows){
                    playerRect.height = frame->original_frame.rows -
                                       playerRect.y;
                }
                p->original_image = frame->original_frame(playerRect);
                p->pos_frame = playerRect;
                p->mask_image = frame->mask_frame(playerRect);
            }
        }

        return players;
    }

    void DPMPlayerExtractor::set_overlapping_threshold(float th){
        if (th < 0.0) th = 0.0; else if (1.0 < th) th = 1.0;
        m_overlap_threshold = th;
    }

    void DPMPlayerExtractor::set_score_threshold(float th){
        m_score_threshold = th;
    }

    float DPMPlayerExtractor::get_overlapping_threshold(){
        return m_overlap_threshold;
    }

    float DPMPlayerExtractor::get_score_threshold(){
        return m_score_threshold;
    }

}