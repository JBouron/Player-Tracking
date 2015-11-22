#ifndef BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H
#define BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H

#include "player_extractor.h"

namespace tmd{

    /**
     * Player extractor using blob detection to extract players from frames.
     */
    class BlobPlayerExtractor : public tmd::PlayerExtractor{
    public:
        virtual std::vector<player_t*> extract_player_from_frame(frame_t*
        frame);
    };
}

#endif //BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H
