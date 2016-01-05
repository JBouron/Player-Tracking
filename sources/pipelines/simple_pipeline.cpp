#include "../../headers/pipelines/simple_pipeline.h"
#include "../../headers/data_structures/frame_t.h"

namespace tmd {
    SimplePipeline::SimplePipeline(std::string video_folder, int camera_index,
                int start_frame, int end_frame, int step_size)
                : Pipeline(video_folder, camera_index, start_frame, end_frame,
                step_size) {

        m_bgSubstractor = new BGSubstractor(video_folder, camera_index,
                                            start_frame, end_frame, step_size);

        if (tmd::Config::use_dpm_player_extractor){
            m_playerExtractor = new DPMPlayerExtractor();
        }
        else {
            m_playerExtractor = new BlobPlayerExtractor();
        }

        m_featuresComparator = new FeatureComparator
                (tmd::Config::features_comparator_center_count,
                 tmd::Config::features_comparator_sample_cols,
                             FeatureComparator::readCentersFromFile());
        m_featuresExtractor = new FeaturesExtractor();
    }

    SimplePipeline::~SimplePipeline() {
        delete m_bgSubstractor;
        delete m_playerExtractor;
        delete m_featuresExtractor;
        delete m_featuresComparator;
    }

    frame_t *SimplePipeline::next_frame() {
        frame_t *frame = m_bgSubstractor->next_frame();
        if (frame == NULL) {
            return NULL;
        }

        if (!tmd::Config::use_bgs){
            const int rows = frame->original_frame.rows;
            const int cols = frame->original_frame.cols;
            frame->mask_frame = cv::Mat::ones(rows, cols, CV_8U);
            frame->colored_mask_frame = frame->original_frame;
            cv::Rect blob = cv::Rect(0, 0, rows, cols);
            frame->blobs.clear();
            frame->blobs.push_back(blob);
        }

        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        extract_players_from_frame(frame);

        return frame;
    }

    void SimplePipeline::set_bgs_properties(float threshold, int history_size,
                                            float learning_rate) {
        m_bgSubstractor->set_threshold_value(threshold);
        m_bgSubstractor->set_history_size(history_size);
        m_bgSubstractor->set_learning_rate(learning_rate);
    }

    void SimplePipeline::extract_players_from_frame(tmd::frame_t *frame) {
        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        tmd::debug("SimplePipeline", "next_frame",std::to_string(players.size())
                                              + " players/blobs extracted.");

        if (!tmd::Config::use_dpm_player_extractor && tmd::Config::use_bgs){
            tmd::debug("SimplePipeline", "next_frame", "Separate blobs.");
            players = BlobSeparator::separate_blobs(players);
            tmd::debug("SimplePipeline", "next_frame", "Done");
        }

        tmd::debug("SimplePipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                       std::to_string(players.size()) + " players detected");
        m_featuresExtractor->extractFeaturesFromPlayers(players);
        m_featuresComparator->detectTeamForPlayers(players);
        frame->players = players;
    }
}