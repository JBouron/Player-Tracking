#include "../../../headers/players_extraction/dpm_based_extraction/dpm_player_extractor.h"
#include "../../../headers/data_structures/frame_t.h"

namespace tmd {
    DPMPlayerExtractor::DPMPlayerExtractor() {
        m_detector = new tmd::DPM();
    }

    DPMPlayerExtractor::~DPMPlayerExtractor() {
        delete m_detector;
    }

    std::vector<player_t *> DPMPlayerExtractor::extract_player_from_frame(
            frame_t *frame) {
        return m_detector->extract_players_and_body_parts(frame);
    }

    void DPMPlayerExtractor::set_overlapping_threshold(float th){
        tmd::Config::dpm_extractor_overlapping_threshold = th;
        recreate_detector();
    }

    void DPMPlayerExtractor::set_score_threshold(float th){
        tmd::Config::dpm_extractor_score_threshold = th;
        recreate_detector();
    }

    float DPMPlayerExtractor::get_overlapping_threshold(){
        return tmd::Config::dpm_extractor_overlapping_threshold;
    }

    float DPMPlayerExtractor::get_score_threshold(){
        return tmd::Config::dpm_extractor_score_threshold;
    }

    void DPMPlayerExtractor::recreate_detector(){
        delete m_detector;
        m_detector = new tmd::DPM();
    }
}