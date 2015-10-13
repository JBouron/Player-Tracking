#include "../headers/calibration_tool.h"
#include "../headers/frame_t.h"

namespace tmd{
    CalibrationTool::CalibrationTool(std::string video_folder_path){
        m_video_folder = video_folder_path;
        for (int i = 0; i < 8; i ++){
            std::string path = video_folder_path + "/ace_" + std::to_string(i) + ".mp4";
            m_videos[i] = new cv::VideoCapture(path);
            if (m_videos[i] == NULL || !m_videos[i]->isOpened()){
                throw std::invalid_argument("Error in CalibrationTool constructor, couldn't open video file : " + path);
            }

            m_bgs[i] = new BGSubstractor(m_videos[i], i);

            m_params[i][THRESHOLD_IDX] = 256;
            m_params[i][HISTORY_SIZE_IDX] = 500;
            m_params[i][LEARNING_RATE_IDX] = 0.0;
        }

        m_current_camera = 0;
    }

    CalibrationTool::~CalibrationTool() {
        for (int i = 0 ; i< 8; i ++){
            free(m_videos[i]);
            free(m_bgs[i]);
        }
    }

    void CalibrationTool::calibrate() {
        cv::namedWindow("Calibration Tool - current frame");
        cv::namedWindow("Calibration Tool - mask frame");

        bool done = false;
        int keyboard = 0;

        while (keyboard != TMD_CALIBRATION_TOOL_ESCAPE_CHAR || !done){
            frame_t* frame;
            if (m_bgs[m_current_camera]->has_next_frame()){
                frame = m_bgs[m_current_camera]->next_frame();
                cv::imshow("Calibration Tool - current frame", frame->original_frame);
                cv::imshow("Calibration Tool - mask frame", frame->mask_frame);

                keyboard = cv::waitKey(10);

                if ((char) keyboard == 'q') m_params[m_current_camera][THRESHOLD_IDX] = cv::max(0, m_params[m_current_camera][THRESHOLD_IDX] - 10.0);
                else if ((char) keyboard == 'w') m_params[m_current_camera][THRESHOLD_IDX] = static_cast<float>(m_params[m_current_camera][THRESHOLD_IDX] + 10.0);
                else if ((char) keyboard == 'a') m_params[m_current_camera][HISTORY_SIZE_IDX] = cv::max(0, m_params[m_current_camera][HISTORY_SIZE_IDX] - 50);
                else if ((char) keyboard == 's') m_params[m_current_camera][HISTORY_SIZE_IDX] = m_params[m_current_camera][HISTORY_SIZE_IDX] + 50;
                else if ((char) keyboard == 'y') m_params[m_current_camera][LEARNING_RATE_IDX] = cv::max(0, m_params[m_current_camera][LEARNING_RATE_IDX] - 0.1);
                else if ((char) keyboard == 'x') m_params[m_current_camera][LEARNING_RATE_IDX] = static_cast<float>(m_params[m_current_camera][LEARNING_RATE_IDX] + 0.1);
                else if ((char) keyboard == ' ') m_current_camera ++;

                if (m_current_camera >= 8){
                    done = true;
                }
            }
        }
    }

    float** CalibrationTool::retrieve_params() {
        return (float**) m_params;
    }
}