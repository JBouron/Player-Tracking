//
// Created by nicolas on 28.11.15.
//

#include <bits/stringfwd.h>
#include "../headers/training_set_creator.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_separator.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"

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
        m_featuresComparator = new FeatureComparator(2, 180, FeatureComparator::readCentersFromFile(2, 180));

        m_camera_index = camera_index;
        m_mask_path = static_mask_path;
        cv::Mat mask = cv::imread(static_mask_path, 0);
        m_bgSubstractor = new BGSubstractor(m_video, mask, camera_index);

        m_dpm = dpm;

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

    frame_t* TrainingSetCreator::next_frame() {

        for (int i = 0; i < m_step - 1; i++) {
            delete m_bgSubstractor->next_frame();
        }

        frame_t *frame = m_bgSubstractor->next_frame();
        if (frame == NULL) {
            return NULL;
        }

        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        cv::Mat coloredMask = get_colored_mask_for_frame(frame);
        frame->original_frame.release();
        frame->original_frame = coloredMask;

        if (!m_dpm) {
            tmd::debug("Pipeline", "next_frame", "Separate blobs.");
            players = BlobSeparator::separate_blobs(players);
            tmd::debug("Pipeline", "next_frame", "Done");
        }

        tmd::debug("Pipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                                             std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        m_featuresComparator->detectTeamForPlayers(players);

        CvScalar torso_color;
        torso_color.val[0] = 255;
        torso_color.val[1] = 255;
        torso_color.val[2] = 0;
        torso_color.val[3] = 255;
        size_t player_count = players.size();
        for (int i = 0; i < player_count; i++) {
            player_t *p = players[i];
            if(p->features.body_parts.size() != 0) {
                m_featuresComparator->addPlayerFeatures(p);
            }
            free_player(players[i]);
        }
        coloredMask.release();
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

    cv::Mat TrainingSetCreator::get_colored_mask_for_frame(frame_t *frame) {
        cv::Mat resulting_image;
        frame->original_frame.copyTo(resulting_image);
        cv::Vec3b black;
        black.val[0] = 0;
        black.val[1] = 0;
        black.val[2] = 0;
        for (int c = 0; c < frame->mask_frame.cols; c++) {
            for (int r = 0; r < frame->mask_frame.rows; r++) {
                if (frame->mask_frame.at<uchar>(r, c) < 127) {
                    resulting_image.at<cv::Vec3b>(r, c) = black;
                }
            }
        }
        return resulting_image;
    }
}

