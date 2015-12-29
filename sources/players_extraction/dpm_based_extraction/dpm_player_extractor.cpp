#include "../../../headers/players_extraction/dpm_based_extraction/dpm_player_extractor.h"

namespace tmd {
    DPMPlayerExtractor::DPMPlayerExtractor() {
        std::vector<std::string> model_files;
        model_files.push_back(tmd::Config::model_file_path);
        m_detector = new cv::LatentSvmDetector();
        bool model_loaded = m_detector->load(model_files);
        if (!model_loaded) {
            throw std::invalid_argument("Error in DPMPlayerExtractor "
                                                "constructor, couldn't load model file.");
        }
        m_overlap_threshold = Config::dpm_extractor_overlapping_threshold;
        if (m_overlap_threshold < 0.0) m_overlap_threshold = 0.0;
        else if (1.0 < m_overlap_threshold) m_overlap_threshold = 1.0;

        m_score_threshold = Config::dpm_extractor_score_threshold;
    }

    DPMPlayerExtractor::~DPMPlayerExtractor() {
        m_detector->clear();
        delete m_detector;
    }

    int get_intersection_area(cv::Rect rect1, cv::Rect rect2) {
        cv::Rect intersection = rect1 & rect2;
        return intersection.area();
    }

    float max(float a, float b) {
        return a > b ? a : b;
    }

    bool is_duplicate(cv::Rect rect,
                      std::vector<cv::LatentSvmDetector::ObjectDetection> &
                      results) {
        size_t vector_size = results.size();
        for (size_t i = 0; i < vector_size; i++) {
            if (rect != results[i].rect) {
                int inter_area = get_intersection_area(rect, results[i].rect);
                float ratio = max((float) inter_area / (float) rect.area(),
                                  (float) inter_area / (float) results[i].rect.area());
                if (ratio > Config::dpm_extractor_duplicate_area_threshold)
                    return true;
            }
        }
        return false;
    }

    std::vector<player_t *> DPMPlayerExtractor::extract_player_from_frame(
            frame_t *frame) {
        cv::Mat image = get_colored_mask_for_frame(frame);
        std::vector<cv::LatentSvmDetector::ObjectDetection> results;
        tmd::debug("DPMPlayerExtractor", "extract_player_from_frame", "Call "
                "detect on image");
        m_detector->detect(image, results, m_overlap_threshold, 1);
        tmd::debug("DPMPlayerExtractor", "extract_player_from_frame", "Done");

        std::vector<tmd::player_t *> players;
        std::vector<cv::LatentSvmDetector::ObjectDetection> filtered_results;

        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].score > m_score_threshold && !is_duplicate
                    (results[i].rect, filtered_results)) {
                filtered_results.push_back(results[i]);
                players.push_back(new player_t);
                tmd::player_t *p = players[players.size() - 1];
                p->likelihood = results[i].score;
                p->frame_index = static_cast<int> (frame->frame_index);

                cv::Rect playerRect = results[i].rect;
                /* Correction. */
                playerRect.height += 40;
                playerRect.width += 40;
                playerRect.x -= 20;
                playerRect.y -= 20;
                /* ** */
                if (playerRect.x < 0) playerRect.x = 0;
                if (playerRect.y < 0) playerRect.y = 0;
                if (playerRect.x + playerRect.width > frame->original_frame
                        .cols) {
                    playerRect.width = frame->original_frame.cols -
                                       playerRect.x;
                }
                if (playerRect.y + playerRect.height > frame->original_frame
                        .rows) {
                    playerRect.height = frame->original_frame.rows -
                                        playerRect.y;
                }
                p->original_image = frame->original_frame(playerRect);
                p->pos_frame = playerRect;
                p->mask_image = frame->mask_frame(playerRect);
            }
        }
        image.release();
        return players;
    }

    void DPMPlayerExtractor::set_overlapping_threshold(float th) {
        if (th < 0.0) th = 0.0; else if (1.0 < th) th = 1.0;
        m_overlap_threshold = th;
    }

    void DPMPlayerExtractor::set_score_threshold(float th) {
        m_score_threshold = th;
    }

    float DPMPlayerExtractor::get_overlapping_threshold() {
        return m_overlap_threshold;
    }

    float DPMPlayerExtractor::get_score_threshold() {
        return m_score_threshold;
    }
}