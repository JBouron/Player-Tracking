#ifndef BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H

#include <set>
#include <iostream>
#include "../../misc/config.h"
#include "../player_extractor.h"

namespace tmd{

    /**
     * Player extractor using blob detection to extract players from frames.
     */
    class BlobPlayerExtractor : public tmd::PlayerExtractor{
    public:
        virtual std::vector<player_t*> extract_player_from_frame(frame_t*
        frame);

    private :
        bool clamp(int rows, int cols, int row, int col);
    };
}

#endif //BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H
