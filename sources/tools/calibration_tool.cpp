#include "../../headers/tools/calibration_tool.h"

/**
 * Tool used during development. Not accessible by the user.
 * May even be a little outdated.
 */

namespace tmd {
    CalibrationTool::CalibrationTool(std::string video_folder_path,
                                     std::string mask_foler_path) {
        m_video_folder = video_folder_path;
        for (int i = 0; i < 8; i++) {
            std::string path =
                    video_folder_path + "/ace_" + std::to_string(i) + ".mp4";
            m_videos[i] = new cv::VideoCapture(path);
            if (m_videos[i] == NULL || !m_videos[i]->isOpened()) {
                throw std::invalid_argument(
                        "Error in CalibrationTool constructor, couldn't open "
                                "video file : " + path);
            }

            std::string mask_path =
                    mask_foler_path + "mask_ace" + std::to_string(i) + ".jpg";

            m_bgs[i] = new BGSubstractor(path, i);

            m_params[i][THRESHOLD_IDX] = 256;
            m_params[i][HISTORY_SIZE_IDX] = 500;
            m_params[i][LEARNING_RATE_IDX] = 0.0;
        }

        m_current_camera = 0;
    }

    CalibrationTool::~CalibrationTool() {
        for (int i = 0; i < 8; i++) {
            free(m_videos[i]);
            free(m_bgs[i]);
        }
    }

    void CalibrationTool::calibrate() {
        cv::namedWindow("Calibration Tool - current frame");
        cv::namedWindow("Calibration Tool - mask frame");

        bool done = false;
        int keyboard = 0;

        while (keyboard != Config::calibration_tool_escape_char && !done) {
            frame_t *frame;
            frame = m_bgs[m_current_camera]->next_frame();
            if (frame != NULL) {
                std::string infos =
                        "Camera " + std::to_string(m_current_camera) +
                        "    Threshold : " + std::to_string(
                                m_params[m_current_camera][THRESHOLD_IDX]) +
                        "   History size : " + std::to_string(
                                m_params[m_current_camera][HISTORY_SIZE_IDX]) +
                        "   Learning rate : " + std::to_string(
                                m_params[m_current_camera][LEARNING_RATE_IDX]);
                cv::putText((frame->mask_frame), infos.c_str(),
                            cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 0, 0));
                cv::rectangle((frame->original_frame), cv::Point(10, 2),
                              cv::Point(800, 20), cv::Scalar(255, 255, 255),
                              -1);
                cv::putText((frame->original_frame), infos.c_str(),
                            cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 0, 0));
                cv::imshow("Calibration Tool - current frame",
                           (frame->original_frame));
                cv::imshow("Calibration Tool - mask frame",
                           (frame->mask_frame));

                m_bgs[m_current_camera]->set_threshold_value(
                        m_params[m_current_camera][THRESHOLD_IDX]);
                m_bgs[m_current_camera]->set_history_size(
            static_cast<int> (m_params[m_current_camera][HISTORY_SIZE_IDX]));
                m_bgs[m_current_camera]->set_learning_rate(
                        m_params[m_current_camera][LEARNING_RATE_IDX]);
                keyboard = cv::waitKey(10);

                if ((char) keyboard == 'q')
                    m_params[m_current_camera][THRESHOLD_IDX] = max(0.f,
                            m_params[m_current_camera][THRESHOLD_IDX] - 10.0);
                else if ((char) keyboard == 'w')
                    m_params[m_current_camera][THRESHOLD_IDX]=static_cast<float>
                        (m_params[m_current_camera][THRESHOLD_IDX] + 10.0);
                else if ((char) keyboard == 'a')
                    m_params[m_current_camera][HISTORY_SIZE_IDX] = max(0.f,
                           m_params[m_current_camera][HISTORY_SIZE_IDX] - 50);
                else if ((char) keyboard == 's')
                    m_params[m_current_camera][HISTORY_SIZE_IDX] =
                            m_params[m_current_camera][HISTORY_SIZE_IDX] + 50;
                else if ((char) keyboard == 'y')
                    m_params[m_current_camera][LEARNING_RATE_IDX] = max(-1.f,
                        m_params[m_current_camera][LEARNING_RATE_IDX] - 0.1);
                else if ((char) keyboard == 'x')
                    m_params[m_current_camera][LEARNING_RATE_IDX] = min(
                            m_params[m_current_camera][LEARNING_RATE_IDX] + 0.1,
                            1.f);
                else if ((char) keyboard == ' ') m_current_camera++;

                if (m_current_camera >= 8) {
                    done = true;
                }

                frame->original_frame.release();
                frame->mask_frame.release();
                free(frame);
            }
        }

        cv::destroyWindow("Calibration Tool - current frame");
        cv::destroyWindow("Calibration Tool - mask frame");
    }

    float **CalibrationTool::retrieve_params() {
        float **params = (float **) calloc(8, sizeof(float *));
        for (int i = 0; i < 8; i++) {
            params[i] = (float *) calloc(3, sizeof(float));
            params[i][0] = m_params[i][0];
            params[i][1] = m_params[i][1];
            params[i][2] = m_params[i][2];
        }
        return params;
    }

    float CalibrationTool::max(double a, double b) {
        if (a > b) return a;
        else return b;
    }

    float CalibrationTool::min(double a, double b) {
        if (a < b) return a;
        else return b;
    }
}