#include <set>
#include "../headers/blob_player_extractor.h"
#include "../headers/player_t.h"
#include "../headers/frame_t.h"
#include "../headers/box_t.h"

#define BUFFER_SIZE 9 //MUST BE ODD
#define MIN_BLOB_SIZE 50 //USED TO FILTER BALL SIZE AND NOISE

using namespace cv;

namespace tmd {
    std::vector<player_t *> BlobPlayerExtractor::extract_player_from_frame(tmd::frame_t *frame) {

        Mat maskImage;
        frame->mask_frame.copyTo(maskImage);
        int rows = maskImage.rows;
        int cols = maskImage.cols;

        unsigned char currentLabel = 1;

        Mat labels;
        labels = Mat::zeros(rows, cols, CV_8UC1);
        unsigned char smallestLabel;
        unsigned char label;
        unsigned char maxLabel = std::numeric_limits<unsigned char>::max();

        std::map<unsigned char, std::set<unsigned char>> labelMap;


        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (maskImage.at<uchar>(row, col) != 0) {

                    std::set<unsigned char> neighbours;
                    smallestLabel = maxLabel;

                    for (int bufferCol = -BUFFER_SIZE / 2; bufferCol <= BUFFER_SIZE / 2; bufferCol++) {
                        for (int bufferRow = -BUFFER_SIZE / 2; bufferRow <= BUFFER_SIZE / 2; bufferRow++) {
                            if (clamp(rows, cols, row + bufferRow, col + bufferCol)
                                && labels.at<uchar>(row, col) != 0) {
                                neighbours.insert(labels.at<uchar>(row, col));
                                smallestLabel = label < smallestLabel ? label : smallestLabel;
                            }
                        }
                    }

                    if (neighbours.empty()) {
                        std::set<unsigned char> setTmp;
                        setTmp.insert(currentLabel);
                        labelMap.insert(std::pair<unsigned char, std::set<unsigned char>>(currentLabel, setTmp));
                        labels.at<uchar>(row, col) = currentLabel;
                        currentLabel++;
                    } else {
                        labels.at<uchar>(row, col) = smallestLabel;
                        for (unsigned char tmp : neighbours) {
                            for (unsigned char tmp2 : neighbours) {
                                std::set<unsigned char> setTmp;
                                setTmp.insert(tmp2);
                                labelMap.insert(std::pair<unsigned char, std::set<unsigned char>>(tmp, setTmp));
                            }
                        }
                    }
                }
            }
        }

        std::map<unsigned char, int> blobSizes;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (maskImage.at<uchar>(row, col) != 0) {
                    std::set<unsigned char> set = labelMap[labels.at<uchar>(row, col)];
                    std::set<unsigned char>::iterator iter = set.begin();
                    label = *iter;
                    labels.at<uchar>(row, col) = label;
                    int currentSize = blobSizes[label] + 1;
                    blobSizes.insert(std::pair<unsigned char, int>(label, currentSize));
                }
            }
        }

        std::vector<player_t *> players;
        for (auto iterator = blobSizes.begin(); iterator != blobSizes.end(); iterator++) {
            if (iterator->second >= MIN_BLOB_SIZE) {
                player_t *player = new player_t;
                label = iterator->first;
                int minRow = std::numeric_limits<int>::max();
                int minCol = std::numeric_limits<int>::max();
                int maxRow = std::numeric_limits<int>::min();
                int maxCol = std::numeric_limits<int>::min();
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        if(labels.at<uchar>(row, col) == label){
                            if(row < minRow){
                                minRow = row;
                            }
                            if(col < minCol){
                                minCol = col;
                            }
                            if(row > maxRow){
                                maxRow = row;
                            }
                            if(col > maxCol){
                                maxCol = col;
                            }
                        }
                    }
                }
                cv::Rect myRect(minCol, minRow, maxCol - minCol, maxRow - minRow);
                player->mask_image = frame->mask_frame.clone()(myRect);
                player->pos_frame = myRect;
                player->original_image = frame->original_frame.clone()(myRect);
                player->frame_index = frame->frame_index;
                players.push_back(player);
            }
        }
        return std::vector<player_t *>();
    }

    bool BlobPlayerExtractor::clamp(int rows, int cols, int row, int col) {
        return row < 0 || row >= rows || col < 0 || col >= cols;
    }

}