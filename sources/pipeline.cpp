#include "../headers/pipeline.h"
#include "../headers/dpm_player_extractor.h"
#include "../headers/debug.h"
#include "../headers/blob_player_extractor.h"
#include "../headers/frame_t.h"
#include "../headers/player_t.h"
#include "../headers/features_t.h"

namespace tmd {
    Pipeline::Pipeline(std::string video_path, std::string static_mask_path,
                       unsigned char camera_index, std::string model_file,
                       bool dpm, bool save_frames, std::string output_folder) {
        m_video_path = video_path;

        m_video = new cv::VideoCapture;
        m_video->open(video_path);

        if (!m_video->isOpened()) {
            throw std::invalid_argument("Error couldn't load the video in the"
                                                " pipeline.");
        }

        m_camera_index = camera_index;
        cv::Mat mask = cv::imread(static_mask_path,0);
        m_bgSubstractor = new BGSubstractor(m_video, mask, camera_index);

        if (dpm) {
            m_playerExtractor = new DPMPlayerExtractor(model_file);
        }
        else {
            m_playerExtractor = new BlobPlayerExtractor();
        }

        m_featuresComparator = new FeatureComparator(2, 180,
                             FeatureComparator::readCentersFromFile(2, 180));

        m_featuresExtractor = new FeaturesExtractor("./res/xmls/person.xml");

        m_running = false;
        m_using_dpm = dpm;
        m_start = 0;
        m_step = 1;
        m_end = -1;

        m_save = save_frames;
        m_output_folder = output_folder;
    }

    Pipeline::~Pipeline() {
        delete m_video;
        delete m_bgSubstractor;
        delete m_playerExtractor;
        delete m_featuresExtractor;
        delete m_featuresComparator;
    }

    void show_body_parts(cv::Mat image, tmd::player_t* p) {
        std::vector<cv::Rect> parts = p->features.body_parts;
        CvScalar color;
        color.val[0] = 255;
        color.val[1] = 0;
        color.val[2] = 255;
        color.val[3] = 255;
        CvScalar torso;
        torso.val[0] = 255;
        torso.val[1] = 255;
        torso.val[2] = 0;
        torso.val[3] = 255;
        const int thickness = 1;
        const int line_type = 8; // 8 connected line.
        const int shift = 0;
        for (int i = 0; i < parts.size(); i++) {
            CvRect r;
            r.x = parts[i].x + p->pos_frame.x;
            r.y = parts[i].y + p->pos_frame.y;
            r.width = parts[i].width;
            r.height = parts[i].height;
            cv::rectangle(image, r, color, thickness, line_type, shift);
        }
    }

    cv::Mat get_colored_mask_for_frame(frame_t* frame){
        cv::Mat resulting_image;
        frame->original_frame.copyTo(resulting_image);
        cv::Vec3b black;
        black.val[0] = 0;
        black.val[1] = 0;
        black.val[2] = 0;
        for (int c = 0; c < frame->mask_frame.cols; c++) {
            for (int r = 0; r < frame->mask_frame.rows; r++) {
                if (frame->mask_frame.at<uchar>(r, c) == 0) {
                    resulting_image.at<cv::Vec3b>(r, c) = black;
                }
            }
        }
        return resulting_image;
    }

    frame_t* Pipeline::next_frame() {
        m_running = true;

        for (int i = 0; i < m_step - 1; i++) {
            delete m_bgSubstractor->next_frame();
        }

        frame_t *frame = m_bgSubstractor->next_frame();
        if (frame == NULL) {
            return NULL;
        }

        cv::imwrite(m_output_folder + "/original_frames/frame" +
                    std::to_string((int) frame->frame_index) + ".jpg",
                frame->original_frame);

        std::vector<tmd::player_t *> players =
                m_playerExtractor->extract_player_from_frame(frame);

        frame->original_frame = get_colored_mask_for_frame(frame);

        tmd::debug("Pipeline", "next_frame", "Frame " + std::to_string
                (m_bgSubstractor->get_current_frame_index()) + " : " +
                         std::to_string(players.size()) + " players detected");

        m_featuresExtractor->extractFeaturesFromPlayers(players);

        const int thickness = 1; // Thickness of the box.
        const int line_type = 8; // 8 connected line.
        const int shift = 0;

        m_featuresComparator->detectTeamForPlayers(players);

        CvScalar torso_color;
        torso_color.val[0] = 255;
        torso_color.val[1] = 255;
        torso_color.val[2] = 0;
        torso_color.val[3] = 255;
        size_t player_count = players.size();
        for (int i = 0; i < player_count; i++) {
            player_t* p = players[i];
            cv::rectangle(frame->original_frame, players[i]->pos_frame,
                          get_team_color(players[i]->team), thickness,
                          line_type, shift);
            show_body_parts(frame->original_frame, p);
            cv::Rect torso;
            torso.x = p->pos_frame.x + p->features.torso_pos.x;
            torso.y = p->pos_frame.y + p->features.torso_pos.y;
            torso.width = p->features.torso_pos.width;
            torso.height = p->features.torso_pos.height;
            cv::rectangle(frame->original_frame, torso,
                          torso_color, thickness,
                          line_type, shift);
            std::string file_name = m_output_folder + "/torsos/torso" +
                    std::to_string((int)frame->frame_index + 1) + "-" +
                    std::to_string(i) + ".jpg";
            cv::imwrite(file_name, frame->original_frame(torso));
            delete players[i];
        }

        if (m_save) {
            std::string file_name = "frame" + std::to_string
                    (m_bgSubstractor->get_current_frame_index()) + ".jpg";
            tmd::debug("Pipeline", "next_frame", "Save frame to : " +
                                                 file_name);
            cv::imwrite(m_output_folder+ "/" +file_name,frame->original_frame);
            file_name = "mask" + std::to_string
                    (m_bgSubstractor->get_current_frame_index()) + ".jpg";
            tmd::debug("Pipeline", "next_frame", "Save frame to : " +
                                                 file_name);
            cv::imwrite(m_output_folder+ "/" +file_name,frame->mask_frame);

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
        if (m_using_dpm) {
            DPMPlayerExtractor *playerExtractor = (DPMPlayerExtractor *)
                    m_featuresExtractor;

            playerExtractor->set_overlapping_threshold(overlapping_threshold);
            playerExtractor->set_score_threshold(score_threshold);
        }
    }

    void Pipeline::set_frame_step_size(int step) {
        m_step = step;
    }

    void Pipeline::set_start_frame(int frame_index) {
        if (!m_running) {
            tmd::debug("Pipeline", "set_frame_step_size", "Setting starting "
                                  "frame to " + std::to_string(frame_index));
            m_start = frame_index;
            m_bgSubstractor->jump_to_frame(frame_index);
        }
    }

    void Pipeline::set_end_frame(int frame_index) {
        if (!m_running) {
            m_end = frame_index;
        }
    }
}
