//
// Created by nicolas on 28.11.15.
//

#include <bits/stringfwd.h>
#include "../headers/training_set_creator.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/debug.h"

namespace tmd {
    TrainingSetCreator::TrainingSetCreator(std::string video_path, std::string static_mask_path,
                                           unsigned char camera_index, std::string model_file, bool dpm,
                                           bool save_frames, std::string output_folder) {
        m_video = new cv::VideoCapture;
        m_video->open(video_path);

        if (!m_video->isOpened()) {
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " pipeline.");
        }

        cv::Mat centers;
        m_featuresComparator = new FeatureComparator(2, 180, centers);

        m_camera_index = camera_index;
        m_mask_path = static_mask_path;
        cv::Mat mask = cv::imread(static_mask_path, 0);
        m_bgSubstractor = new BGSubstractor(m_video, mask, camera_index);

        if (dpm) {
            m_playerExtractor = new DPMPlayerExtractor(model_file);
        }
        else {
            m_playerExtractor = new BlobPlayerExtractor();
        }

        m_featuresExtractor = new FeaturesExtractor("./res/xmls/person.xml");
    }

    TrainingSetCreator::~TrainingSetCreator() {

    }

    frame_t*  TrainingSetCreator::next_frame() {

        for (int i = 0; i < m_step - 1; i++) {
            delete m_bgSubstractor->next_frame();
        }

        frame_t *frame = m_bgSubstractor->next_frame();
        if (frame == NULL) {
            return NULL;
        }

        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        tmd::debug("Pipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                                             std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        for(int i = 0; i < players.size(); i++){
            m_featuresComparator->addPlayerFeatures(players[i]);
        }

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
        m_bgSubstractor = new BGSubstractor(m_video, mask, m_camera_index);
    }

    void TrainingSetCreator::write_centers() {
        m_featuresComparator->runClustering();
        m_featuresComparator->writeCentersToFile();
    }

    void TrainingSetCreator::set_frame_step_size(int step) {
        m_step = step;
    }
}

