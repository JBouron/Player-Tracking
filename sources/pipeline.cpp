#include "../headers/pipeline.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/player_t.h"
#include "../headers/frame_t.h"

namespace tmd{
    Pipeline::Pipeline(std::string video_path, unsigned char camera_index,
                       std::string model_file, bool dpm, bool save_frames,
                       std::string output_folder) {
        m_video_path = video_path;
        m_video = new cv::VideoCapture;
        m_video->open(video_path);

        if (!m_video->isOpened()){
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " piepline.");
        }

        m_camera_index = camera_index;
        m_bgSubstractor = new BGSubstractor(m_video, camera_index);

        if (dpm){
            m_playerExtractor = new DPMPlayerExtractor(model_file);
        }
        else{
            // TODO : Create a blob-detection instance.
        }

        m_featuresComparator = new FeatureComparator(2, 180,
                             FeatureComparator::readCentersFromFile(1, 180));

        m_running = false;
        m_using_dpm = dpm;
        m_start = 0;
        m_step = 1;
        m_end = -1;

        m_save = save_frames;
        m_output_folder = output_folder;

        // Take the first frame so that the BGS can build a model for the
        // background.
        delete m_bgSubstractor->next_frame();
    }

    Pipeline::~Pipeline() {
        delete m_video;
        delete m_bgSubstractor;
        delete m_playerExtractor;
        delete m_featuresExtractor;
        delete m_featuresComparator;
    }

    frame_t* Pipeline::next_frame() {
        m_running = true;

        for (int i = 0 ; i < m_step ; i ++){
            delete m_bgSubstractor->next_frame();
        }

        frame_t* frame = m_bgSubstractor->next_frame();
       	if (frame == NULL){
		    return NULL;
	    }

        std::vector<tmd::player_t*> players =
                m_playerExtractor->extract_player_from_frame(frame);

        tmd::debug("Pipeline", "next_frame", "Frame " + std::to_string
            (m_bgSubstractor->get_current_frame_index()) + " : " +
                std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        const int thickness = 1; // Thickness of the box.
        const int line_type = 8; // 8 connected line.
        const int shift = 0;



        size_t player_count = players.size();
        for (int i = 0 ; i < player_count ; i ++){
            cv::Mat result = m_featuresComparator->getClosestCenter(players[i]);
            players[i]->team = get_team_from_center(result);
            tmd::debug("Pipeline", "next_frame", "Player " + std::to_string(i)
                          + " detected with team " + get_team_string
                         (players[i]->team));

            cv::rectangle(frame->original_frame, players[i]->pos_frame,
                          get_team_color(players[i]->team), thickness,
                          line_type, shift);

            delete players[i];
        }
        return frame;
    }

    void Pipeline::set_bgs_properties(float threshold, int history_size,
                                      float learning_rate) {
        m_bgSubstractor->set_threshold_value(threshold);
        m_bgSubstractor->set_history_size(history_size);
        m_bgSubstractor->set_learning_rate(learning_rate);
    }

    void Pipeline::set_dpm_properties(float overlapping_threshold,
                                      float score_threshold) {
        if (m_using_dpm){
            DPMPlayerExtractor *playerExtractor = (DPMPlayerExtractor*)
                    m_featuresExtractor;

            playerExtractor->set_overlapping_threshold(overlapping_threshold);
            playerExtractor->set_score_threshold(score_threshold);
        }
    }

    void Pipeline::set_frame_step_size(int step) {
        m_step = step;
    }

    void Pipeline::set_start_frame(int frame_index) {
        if (!m_running){
            m_start = frame_index;
            m_bgSubstractor->jump_to_frame(m_start);
        }
    }

    void Pipeline::set_end_frame(int frame_index) {
        if (!m_running){
            m_end = frame_index;
        }
    }

    team_t Pipeline::get_team_from_center(cv::Mat closest_center){
        int max_hue_value = 0;
        for (int c = 0 ; c < closest_center.cols ; c ++){
            if (closest_center.at<uchar>(0, c) == 1.0){
                max_hue_value = c;
                break;
            }
        }
        if (TMD_FEATURE_EXTRACTOR_TH_GREEN_LOW <= max_hue_value &&
                max_hue_value <= TMD_FEATURE_EXTRACTOR_TH_GREEN_HIGH){
            return TEAM_A;
        }
        else{
            return TEAM_B;
        }
    }
}
