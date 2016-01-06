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
        /**
         * Extract the blob / players from the given frame and returns them in a
         * vector.
         * Note : At this point we can have blobs containing multiple
         * players, in case of occlusion for example. These blobs thus need
         * to be separated later. (See BlobSeparator).
         */
        virtual std::vector<player_t*> extract_player_from_frame(frame_t*
        frame);

    private :
        /**
         * Clamp the results according to the given limits.
         */
        bool clamp(int rows, int cols, int row, int col);
    };
}

#endif //BACHELOR_PROJECT_BLOB_PLAYER_EXTRACTOR_H
