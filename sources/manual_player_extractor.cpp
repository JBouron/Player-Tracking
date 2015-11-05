#include "../headers/manual_player_extractor.h"
#include "../headers/box_t.h"
#include "../headers/frame_t.h"
#include "../headers/debug.h"
#include "../headers/player_t.h"

namespace tmd{
        bool ManualPlayerExtractor::mFirstClick = true;
        bool ManualPlayerExtractor::mBoxComplete = false;
        std::vector<cv::Rect> ManualPlayerExtractor::mBoxes;

    std::vector<player_t*> ManualPlayerExtractor::extract_player_from_frame(
            frame_t *frame) {
        std::vector<tmd::player_t*> v;
        for (int i = 0 ; i < 3 ; i ++){
            v.push_back(new tmd::player_t);
            v[i]->original_image = cv::imread("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/img" + std::to_string(i+1) +".jpg");
        }

        mFirstClick = true;
        mBoxComplete = false;
        /*cv::namedWindow("Manual player extraction");
        cv::setMouseCallback("Manual player extraction", onMouseClick, NULL);*/
        int keyboard = 0;
        cv::Mat image;
            image = frame->original_frame.clone();
        while ((char) keyboard != 32){ // Wait space to got ot next frame
            size_t max_idx;
            if (mBoxComplete) max_idx = mBoxes.size();
            else{
                if (mBoxes.size() == 0) max_idx = 0; // size_t should not be -1.
                else max_idx = mBoxes.size() - 1;
            }
            for (size_t i = 0; i < max_idx; i ++){
                cv::rectangle(image, mBoxes[i], cv::Scalar(255, 0, 0));
            }
            //cv::imshow("Manual player extraction", image);
            keyboard = cv::waitKey(15);
        }
        //cv::destroyAllWindows();
        std::vector<player_t*> players;
        for (size_t i = 0 ; i < mBoxes.size() ; i ++){
            tmd::debug("boxes");
            cv::Mat cpy = frame->original_frame.clone();
            players.push_back(new player_t);
            players[i]->original_image = cpy(mBoxes[i]);
            cv::imwrite("/home/jbouron/EPFL/BA5/PlayfulVision/Bachelor-Project/misc/images/player" + std::to_string(i) +".jpg", players[i]->original_image);
            //players[i]->mask_image = (*frame->mask_frame)(mBoxes[i]);
            players[i]->frame_index = frame->frame_index;
        }
        return v;
        //return players;
    }

    void ManualPlayerExtractor::onMouseClick(int event, int x, int y, int f, void* d){
        size_t len = mBoxes.size();
        if  ( event == cv::EVENT_LBUTTONDOWN ) {
            if (len == 0 || mBoxComplete){
                len ++;
                mBoxes.push_back(cv::Rect(0, 0, 0, 0));
            }

            tmd::debug("ManualPlayerExtractor", "onMouseClick", "len = " + std::to_string(len));

            if (mFirstClick){
                tmd::debug("ManualPlayerExtractor", "onMouseClick", "First click : " + std::to_string(x) + "  "  + std::to_string(y));
                mBoxes[len-1].x = x;
                mBoxes[len-1].y = y;
                mBoxComplete = false;
            }
            else{
                int dx = x - mBoxes[len-1].x;
                int dy = y - mBoxes[len-1].y;
                tmd::debug("ManualPlayerExtractor", "onMouseClick", "Second click : " + std::to_string(x) + "  "  + std::to_string(y) + "  " + std::to_string(dx) + "  "  + std::to_string(dy));

                if (dx < 0){
                    mBoxes[len-1].x = x;
                    mBoxes[len-1].width = -dx;
                }
                else{
                    mBoxes[len-1].width = dx;
                }

                if (dy < 0){
                    mBoxes[len-1].y = y;
                    mBoxes[len-1].height = -dy;
                }
                else{
                    mBoxes[len-1].height = dy;
                }
                mBoxComplete = true;
            }
            mFirstClick = !mFirstClick;
        }
    }
}