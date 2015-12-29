#ifndef BACHELOR_PROJECT_BLOB_SEPARATOR_H
#define BACHELOR_PROJECT_BLOB_SEPARATOR_H

#include <vector>
#include "../dpm_based_extraction/dpm_player_extractor.h"
#include <opencv2/objdetect/objdetect.hpp>
#include "../../pipelines/pipeline.h"

namespace tmd{
    class BlobSeparator{
    public:
        static std::vector<tmd::player_t*> separate_blobs
                (std::vector<tmd::player_t*> players);
    };
}

#endif //BACHELOR_PROJECT_BLOB_SEPARATOR_H