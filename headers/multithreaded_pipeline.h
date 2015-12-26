#include "pipeline.h"
#include "pipeline_thread.h"

#ifndef BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H
#define BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H

namespace tmd{
    /**
     * Class reprensenting a multithreaded pipeline.
     */
    class MultithreadedPipeline : public Pipeline{

    public:
        /**
         * Constructor of the Multithreaded Pipeline.
         * video_path : Path to the video on which the pipeline will operate.
         * thread_count : number of thread to use.
         * model_file : Path to the file containing the model.
         * save_frames : When asserted the frame will be saved i the
         * output_folder.
         * output_folder : Folder that will contain thew saved frames (if
         * save_frames is asserted).
         */
        MultithreadedPipeline(std::string video_folder, int camera_index, int
                thread_count, int start_frame, int end_frame, int step_size);

        ~MultithreadedPipeline();

        /**
         * Extract the next frame from the input video.
         * Run the full pipeline on this frame.
         * And finally return the result.
         *
         * If there is no frame left the method simply returns NULL.
         *
         * Note that the user has to take care of freeing the frames.
         */
        frame_t* next_frame();

    private:
        void create_threads();

        tmd::PipelineThread** m_pipeline_threads;
        bool m_threads_ready;
        int m_thread_count;
        int m_frame_pos;
        int m_next_thread_to_use; // Keep track of the next thread we should
        // use to get the next frame.
    };
}

#endif //BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H
