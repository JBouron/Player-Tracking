#include "../headers/simple_pipeline.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_separator.h"
#include "../headers/frame_t.h"

namespace tmd {
    SimplePipeline::SimplePipeline(std::string video_path,
                                   std::string model_file, int
                                   start_frame, int end_frame, int step_size) : Pipeline
                          (video_path, model_file, start_frame, end_frame, step_size){
        // TODO : Remove place holders.
        std::string static_mask_path = "./res/bgs_masks/mask_ace0.jpg";
        cv::Mat mask = cv::imread(static_mask_path, 0);
        m_bgSubstractor = new BGSubstractor(video_path, mask, 0, start_frame,
                                            step_size);

        m_playerExtractor = new BlobPlayerExtractor();

        m_featuresComparator = new FeatureComparator(2, 180,
                                                     FeatureComparator::readCentersFromFile(2, 180));

        m_featuresExtractor = new FeaturesExtractor("./res/xmls/person.xml");
    }

    SimplePipeline::~SimplePipeline() {
        delete m_bgSubstractor;
        delete m_playerExtractor;
        delete m_featuresExtractor;
        delete m_featuresComparator;
    }

    frame_t* SimplePipeline::next_frame(){
        m_running = true;

        frame_t* frame = fetch_next_frame();

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

    tmd::frame_t* SimplePipeline::fetch_next_frame(){
        for (int i = 0; i < m_step - 1; i++) {
            // Throwing away the frames that are not important for us.
            delete m_bgSubstractor->next_frame();
        }
        frame_t *frame = m_bgSubstractor->next_frame();
        return frame;
    }

    void SimplePipeline::extract_players_from_frame
            (tmd::frame_t* frame){
        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        tmd::debug("SimplePipeline", "next_frame", std::to_string(players.size()) +
                                             " players/blobs extracted.");

        cv::Mat coloredMask = get_colored_mask_for_frame(frame);
        frame->original_frame.release();
        frame->original_frame = coloredMask;

        tmd::debug("SimplePipeline", "next_frame", "Separate blobs.");
        players = BlobSeparator::separate_blobs(players);
        tmd::debug("SimplePipeline", "next_frame", "Done");

        tmd::debug("SimplePipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                                             std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        m_featuresComparator->detectTeamForPlayers(players);
        coloredMask.release();
        frame->players = players;
    }
}