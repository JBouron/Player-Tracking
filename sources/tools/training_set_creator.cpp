#include "../../headers/tools/training_set_creator.h"

namespace tmd {
    TrainingSetCreator::TrainingSetCreator(std::string video_folder, int camera_index, int start_frame,
                                           int end_frame, int step_size) {
        m_bgSubstractor = new BGSubstractor(video_folder, camera_index, start_frame, end_frame, step_size);
        m_playerExtractor = new BlobPlayerExtractor();
        cv::Mat centers;
        m_featuresComparator = new FeatureComparator(2, 180, centers);
        m_featuresExtractor = new FeaturesExtractor();
    }

    TrainingSetCreator::~TrainingSetCreator() {
        delete m_bgSubstractor;
        delete m_playerExtractor;
        delete m_featuresExtractor;
        delete m_featuresComparator;
    }

    frame_t *TrainingSetCreator::next_frame() {
        frame_t *frame = m_bgSubstractor->next_frame();
        if (frame == NULL) {
            return NULL;
        }

        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        extract_players_from_frame(frame);
        return frame;
    }

    void TrainingSetCreator::set_new_video_path(std::string video_path) {
        delete m_video;
        m_video = new cv::VideoCapture;
        m_video->open(video_path);

        if (!m_video->isOpened()) {
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " pipeline.");
        }

        cv::Mat mask = cv::imread(m_mask_path, 0);
        delete m_bgSubstractor;
        m_bgSubstractor = new BGSubstractor(video_path, m_camera_index);
    }

    void TrainingSetCreator::write_centers(int frame_index) {
        m_featuresComparator->runClustering();
        m_featuresComparator->writeCentersToFile(frame_index);
    }

    void TrainingSetCreator::write_centers() {
        m_featuresComparator->runClustering();
        m_featuresComparator->writeCentersToFile();
    }

    void TrainingSetCreator::extract_players_from_frame(tmd::frame_t *frame) {
        tmd::debug("SimplePipeline", "next_frame", "Extracting players.");
        std::vector<tmd::player_t *> players = m_playerExtractor->extract_player_from_frame(frame);
        tmd::debug("SimplePipeline", "next_frame", std::to_string(players.size()) + " players/blobs extracted.");

        cv::Mat coloredMask = get_colored_mask_for_frame(frame);
        frame->original_frame.release();
        frame->original_frame = coloredMask;

        tmd::debug("SimplePipeline", "next_frame", "Separate blobs.");
        players = BlobSeparator::separate_blobs(players);
        tmd::debug("SimplePipeline", "next_frame", "Done");

        tmd::debug("SimplePipeline", "next_frame", "Frame " +
                                                   std::to_string(m_bgSubstractor->get_current_frame_index()) + " : " +
                                                   std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        size_t player_count = players.size();
        for (int i = 0; i < player_count; i++) {
            player_t *p = players[i];
            if (p->features.body_parts.size() != 0) {
                m_featuresComparator->addPlayerFeatures(p);
            }
        }

        coloredMask.release();
        frame->players = players;
    }
}

