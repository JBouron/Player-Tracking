#include "../../../headers/players_extraction/blob_based_extraction/blob_player_extractor.h"
#include "../../../headers/data_structures/frame_t.h"

using namespace cv;

namespace tmd {
    std::vector<player_t *> BlobPlayerExtractor::extract_player_from_frame(
            tmd::frame_t *frame) {

        Mat maskImage;
        frame->mask_frame.copyTo(maskImage);
        int rows = maskImage.rows;
        int cols = maskImage.cols;

        int currentLabel = 1;

        Mat labels;
        labels = Mat::zeros(rows, cols, CV_32SC1);
        int smallestLabel;
        int label;
        int maxLabel = std::numeric_limits<int>::max();

        std::map<int, std::set<int>> labelMap;
        const int BUFFER_SIZE = Config::blob_player_extractor_buffer_size;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (maskImage.at<uchar>(row, col) != 0) {
                    std::set<int> neighbours;
                    smallestLabel = maxLabel;

                    for (int bufferCol = -BUFFER_SIZE / 2;
                         bufferCol <= BUFFER_SIZE / 2;
                         bufferCol++) {

                        for (int bufferRow = -BUFFER_SIZE / 2;
                             bufferRow <= BUFFER_SIZE / 2;
                             bufferRow++) {

                            if (!clamp(rows, cols, row + bufferRow,
                                       col + bufferCol)) {

                                if (labels.at<int>(row + bufferRow,
                                                   col + bufferCol) != 0) {

                                    label = labels.at<int>(row + bufferRow,
                                                           col + bufferCol);

                                    neighbours.insert(label);

                                    smallestLabel = label < smallestLabel ?
                                                    label : smallestLabel;
                                }
                            }
                        }
                    }

                    if (neighbours.empty()) {
                        std::set<int> setTmp;
                        setTmp.insert(currentLabel);

                        labelMap.insert(std::pair<int,std::set<int>>
                                                (currentLabel, setTmp));

                        labels.at<int>(row, col) = currentLabel;
                        currentLabel++;
                    } else {
                        labels.at<int>(row, col) = smallestLabel;
                        for (int tmp : neighbours) {
                            for (int tmp2 : neighbours) {
                                labelMap[tmp].insert(tmp2);
                            }
                            labelMap[tmp].insert(smallestLabel);
                        }
                        labelMap[smallestLabel].insert(smallestLabel);
                    }
                }
            }
        }

        std::map<int, int> blobSizes;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (maskImage.at<uchar>(row, col) != 0) {
                    std::set<int> set = labelMap[labels.at<int>(row, col)];
                    std::set<int>::iterator iter = set.begin();
                    label = *iter;

                    labels.at<int>(row, col) = label;
                    int currentSize = 0;
                    std::map<int, int>::iterator it = blobSizes.find(label);

                    if (it != blobSizes.end()) {
                        currentSize = it.operator*().second;
                    }

                    currentSize++;
                    blobSizes[label] = currentSize;
                }
            }
        }

        std::vector<player_t *> players;
        for (auto iterator = blobSizes.begin(); iterator != blobSizes.end();
             iterator++) {

            if(iterator->second >= Config::blob_player_extractor_min_blob_size){
                player_t *player = new player_t;
                label = iterator->first;
                int minRow = std::numeric_limits<int>::max();
                int minCol = std::numeric_limits<int>::max();
                int maxRow = std::numeric_limits<int>::min();
                int maxCol = std::numeric_limits<int>::min();
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        if (labels.at<int>(row, col) == label) {
                            if (row < minRow) {
                                minRow = row;
                            }
                            if (col < minCol) {
                                minCol = col;
                            }
                            if (row > maxRow) {
                                maxRow = row;
                            }
                            if (col > maxCol) {
                                maxCol = col;
                            }
                        }
                    }
                }

                int tpX = (minCol - 20) < 0 ? 0 : (minCol - 20);
                int tpY = (minRow - 20) < 0 ? 0 : (minRow - 20);
                int btX = (maxCol + 20) > cols ? cols : (maxCol + 20);
                int btY = (maxRow + 20) > rows ? rows : (maxRow + 20);
                cv::Rect myRect(tpX, tpY, btX - tpX, btY - tpY);
                frame->blobs.push_back(myRect);
                player->mask_image = frame->mask_frame(myRect);
                player->pos_frame = myRect;
                player->team = TEAM_UNKNOWN;
                player->original_image = frame->original_frame(myRect);
                player->frame_index = frame->frame_index;
                players.push_back(player);
            }
        }

        return players;
    }

    bool BlobPlayerExtractor::clamp(int rows, int cols, int row, int col) {
        return row < 0 || row >= rows || col < 0 || col >= cols;
    }

}