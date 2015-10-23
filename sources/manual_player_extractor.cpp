#include "../headers/manual_player_extractor.h"
#include "../headers/box_t.h"
#include "../headers/frame_t.h"

namespace tmd{
        bool ManualPlayerExtractor::mFirstClick = true;
        bool ManualPlayerExtractor::mBoxComplete = false;
        std::vector<tmd::box_t*> ManualPlayerExtractor::mBoxes;

    std::vector<player_t*> ManualPlayerExtractor::extract_player_from_frame(
            frame_t *frame) {
        mFirstClick = true;
        mBoxComplete = false;
        cv::namedWindow("Manual player extraction");
        cv::setMouseCallback("Manual player extraction", onMouseClick, NULL);
        int keyboard = 0;
        cv::Mat image;
        while ((char) keyboard != 32){ // Wait space to got ot next frame
            size_t max_idx;
            if (mBoxComplete) max_idx = mBoxes.size();
            else max_idx = mBoxes.size() - 1;
            image = frame->original_frame->clone();
            for (size_t i = 0; i < max_idx; i ++){
                cv::rectangle(image, cv::Point(mBoxes[i]->pos.x, mBoxes[i]->pos.y), cv::Point(mBoxes[i]->size.x, mBoxes[i]->size.y), cv::Scalar(255, 0, 0));
            }
            cv::imshow("Manual player extraction", image);
            keyboard = cv::waitKey(15);
        }
    }

    void ManualPlayerExtractor::onMouseClick(int event, int x, int y, int f, void* d){
        size_t len = mBoxes.size();
        if  ( event == cv::EVENT_LBUTTONDOWN ) {
            if (len == 0 || mBoxComplete){
                len ++;
                mBoxes.push_back((box_t*) malloc(sizeof(box_t)));
            }

            box_t* b = mBoxes[len-1];

            if (mFirstClick){
                b->pos.x = x;
                b->pos.y = y;
                mBoxComplete = false;
            }
            else{
                int dx = b->pos.x - x;
                int dy = b->pos.y - y;

                if (dx < 0){
                    b->pos.x = x;
                    b->size.x = -dx;
                }
                else{
                    b->size.x = dx;
                }

                if (dy < 0){
                    b->pos.y = y;
                    b->size.y = -dy;
                }
                else{
                    b->size.y = dy;
                }
                mBoxComplete = true;
            }
            mFirstClick = !mFirstClick;
        }
    }
}