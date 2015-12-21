#ifndef BACHELOR_PROJECT_WRITE_BUFFER_H
#define BACHELOR_PROJECT_WRITE_BUFFER_H

#include <string>
#include <atomic>
#include <thread>
#include "frame_t.h"

namespace tmd{
    /**
     * Class representing a write buffer.
     * It holds a data structure containing the frames to be written on the
     * disk. When a certain count of frames reside in the buffer, a thread is
     * responsible to write them on the disk without any intervention from
     * the user.
     */
    class WriteBuffer{
    public:
        /**
         * Constructor of the buffer :
         * _ dest_folder : The folder in which we will write the frames.
         * _ buffer_max_size : The maximum size of the buffer.
         * _ save_maks : When asserted the mask of the frames will be saved
         * too.
         */
        WriteBuffer(std::string dest_folder, size_t buffer_max_size, bool
        save_mask);

        ~WriteBuffer();

        /**
         * Add a frame to the buffer.
         * The frame is copied from the original one.
         */
        void add_to_buffer(tmd::frame_t *frame);

        /**
         * Used when the user decides to write the buffer before exiting.
         */
        void force_write();
    private:
        /**
         * Flush the buffer and write its content on the disk.
         */
        void flush_buffer();

        std::vector<tmd::frame_t*> m_buffer;
        size_t m_buffer_size;       // The current size of the buffer.
        size_t m_buffer_max_size;   // The maximum size of the buffer.
        size_t m_elements_count;    // The number of elements written on disk.
                                    // Used to name files.
        std::atomic<bool> m_write_flag;   // Is the buffer currently written in
                                    // memory by another thread ?.
        std::string m_dest_folder;
        bool m_save_mask;
        std::thread m_writing_thread;
    };
}

#endif //BACHELOR_PROJECT_WRITE_BUFFER_H
