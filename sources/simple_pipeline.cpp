#include "../headers/simple_pipeline.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_separator.h"

namespace tmd {
    SimplePipeline::SimplePipeline(std::string video_path,
                                   std::string model_file, bool save_frames,
                                   std::string output_folder) : Pipeline
                          (video_path, model_file, save_frames, output_folder){
        // TODO : Remove place holders.
        std::string static_mask_path = "./res/bgs_masks/mask_ace0.jpg";
        cv::Mat mask = cv::imread(static_mask_path, 0);
        m_bgSubstractor = new BGSubstractor(m_video, mask, 0);

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
        std::vector<tmd::player_t*> players = extract_players_from_frame(frame);

        tmd::debug("SimplePipeline", "next_frame", "Draw frame.");
        const int thickness = 1; // Thickness of the box.
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        CvScalar torso_color;
        torso_color.val[0] = 255;
        torso_color.val[1] = 255;
        torso_color.val[2] = 0;
        torso_color.val[3] = 255;
        size_t player_count = players.size();
        for (int i = 0; i < player_count; i++) {
            player_t *p = players[i];
            cv::rectangle(frame->original_frame, players[i]->pos_frame,
                          get_team_color(players[i]->team), thickness,
                          line_type, shift);

            cv::Rect torso;
            torso.x = p->pos_frame.x + p->features.torso_pos.x;
            torso.y = p->pos_frame.y + p->features.torso_pos.y;
            torso.width = p->features.torso_pos.width;
            torso.height = p->features.torso_pos.height;
            cv::rectangle(frame->original_frame, torso,
                          torso_color, thickness,
                          line_type, shift);
            free_player(players[i]);
        }

        if (m_save) {
            std::string index_string = std::to_string(static_cast<int>(frame->frame_index));
            std::cout << "Write frame " << index_string << std::endl;
            std::string file_name = "frame" + index_string + ".jpg";
            tmd::debug("SimplePipeline", "next_frame", "Save frame to : " +
                                                 file_name);
            cv::imwrite(m_output_folder + "/" + file_name, frame->original_frame);
            /*file_name = "mask" + index_string + ".jpg";
            tmd::debug("Pipeline", "next_frame", "Save mask to : " +
                                                 file_name);
            cv::imwrite(m_output_folder + "/" + file_name, frame->mask_frame);*/
        }
        return frame;
    }

    std::vector<tmd::player_t*> SimplePipeline::next_players(){
        m_running = true;

        frame_t* frame = fetch_next_frame();

        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        return extract_players_from_frame(frame);
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

    std::vector<tmd::player_t*> SimplePipeline::extract_players_from_frame
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
        return players;
    }

    void SimplePipeline::set_frame_step_size(int step) {
        m_step = step;
    }

    void SimplePipeline::set_start_frame(int frame_index) {
        if (!m_running) {
            tmd::debug("Pipeline", "set_frame_step_size", "Setting starting "
                                                                  "frame to " + std::to_string(frame_index));
            m_start = frame_index;
            m_bgSubstractor->jump_to_frame(frame_index);
            tmd::debug("Pipeline", "set_frame_step_size", "Done");
        }
    }

    void SimplePipeline::set_end_frame(int frame_index) {
        if (!m_running) {
            m_end = frame_index;
        }
    }
}