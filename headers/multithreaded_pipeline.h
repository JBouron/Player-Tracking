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
        MultithreadedPipeline(std::string video_path, int thread_count,
                              std::string model_file, bool save_frames = false,
                              std::string output_folder = "");

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


        /**
         * Extract the next frame from the input video.
         * Run the full pipeline on it.
         * Then returns a vector containing all the player for this frame.
         * Along with their team, features, ...
         */
        std::vector<tmd::player_t*> next_players();

        /**
         * Set the step size between to consecutive extracted frames.
         */
        void set_frame_step_size(int step);

        /**
         * Set the starting frame index.
         * The extraction must not begun before this operation.
         */
        void set_start_frame(int frame_index);

        /**
         * Set the frame index
         */
        void set_end_frame(int frame_index);

    private:
        tmd::PipelineThread** m_pipeline_threads;
        bool m_threads_ready;
        int m_thread_count;
    };
}

#endif //BACHELOR_PROJECT_MULTITHREADED_PIPELINE_H
