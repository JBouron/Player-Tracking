#include "../../headers/pipelines/simple_pipeline.h"
#include "../../headers/data_structures/frame_t.h"

namespace tmd {
    SimplePipeline::SimplePipeline(std::string video_folder, int camera_index, int start_frame, int end_frame,
                                   int step_size) : Pipeline(video_folder, camera_index, start_frame,
                                                             end_frame, step_size) {
        m_bgSubstractor = new BGSubstractor(video_folder, camera_index,
                                            start_frame, end_frame, step_size);
        m_playerExtractor = new BlobPlayerExtractor();
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

        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        extract_players_from_frame(frame);

        return frame;
    }

    void SimplePipeline::set_bgs_properties(float threshold, int history_size, float learning_rate) {
        m_bgSubstractor->set_threshold_value(threshold);
        m_bgSubstractor->set_history_size(history_size);
        m_bgSubstractor->set_learning_rate(learning_rate);
    }

    void SimplePipeline::extract_players_from_frame(tmd::frame_t *frame) {
        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        tmd::debug("SimplePipeline", "next_frame", std::to_string(players.size()) +
                                                   " players/blobs extracted.");

        cv::Mat coloredMask = get_colored_mask_for_frame(frame);
        /*frame->colored_mask_frame = coloredMask;*/
        frame->original_frame = coloredMask;

        tmd::debug("SimplePipeline", "next_frame", "Separate blobs.");
        players = BlobSeparator::separate_blobs(players);
        tmd::debug("SimplePipeline", "next_frame", "Done");

        tmd::debug("SimplePipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                                                   std::to_string(players.size()) + " players detected");
        m_featuresExtractor->extractFeaturesFromPlayers(players);
        m_featuresComparator->detectTeamForPlayers(players);
        //coloredMask.release();
        frame->players = players;
    }
}