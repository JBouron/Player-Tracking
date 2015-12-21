#include "../headers/write_buffer.h"
#include "../headers/frame_t.h"
#include "../headers/debug.h"

namespace tmd{
    WriteBuffer::WriteBuffer(std::string dest_folder, size_t buffer_max_size,
                             bool save_mask) {
        m_dest_folder = dest_folder;
        m_buffer_max_size = buffer_max_size;
        m_buffer_size = 0;
        m_save_mask = save_mask;
        m_elements_count = 0;
        m_write_flag = false;
        m_save_mask = false;
    }

    WriteBuffer::~WriteBuffer(){
        if (m_write_flag){
            tmd::debug("WriteBuffer", "~WriteBuffer", "Waiting for wrinting "
                    "thread to finish.");
            m_writing_thread.join();
        }
    }

    void WriteBuffer::add_to_buffer(tmd::frame_t *frame){
        if (m_buffer.size() >= m_buffer_max_size){
            flush_buffer();
        }
        frame_t* copy = new frame_t;
        copy->frame_index = frame->frame_index;
        copy->original_frame = frame->original_frame.clone();
        if (m_save_mask){
            copy->mask_frame = frame->mask_frame.clone();
        }
        m_buffer.push_back(copy);
    }

    void WriteBuffer::force_write(){
        flush_buffer();
        if (m_write_flag){
            m_writing_thread.join();
        }
    }

    void _write_buffer_to_disk(std::vector<frame_t*> frames,
           std::string dest_folder, bool save_mask, size_t* elements_count,
                               std::atomic<bool> &write_flag){
        size_t size = frames.size();
        for (size_t i = 0 ; i < size ; i ++){
            int file_idx = static_cast<int>(frames[i]->frame_index);
            std::string file_name = dest_folder + "/frame" + std::to_string(file_idx) +
                    ".jpg";
            tmd::debug("WriteBuffer", "_write_buffer_to_disk", "Thread : "
                    "writing to file : " + file_name);
            cv::imwrite(file_name, frames[i]->original_frame);
            if (save_mask){
                file_name = dest_folder + "/mask" + std::to_string(file_idx) +
                            ".jpg";
                tmd::debug("WriteBuffer", "_write_buffer_to_disk", "Thread : "
                                           "writing to file : " + file_name);
                cv::imwrite(file_name, frames[i]->mask_frame);
            }
        }
        (*elements_count) += size;
        write_flag = false;
        tmd::debug("WriteBuffer", "_write_buffer_to_disk", "Thread : Writing "
                "done");
    }

    void WriteBuffer::flush_buffer(){
        /** Important note :
         *  There should be only to thread maximum here. The main thread trying
         *  to flush the buffer and the writing thread.
         *  With this assumption we can to the "barrier" easily by just waiting
         *  the write_flag to be false.
         *
         *  TODO : Change the implementation, not really critical though.
         */
        if (m_write_flag){
            tmd::debug("WriteBuffer", "flush_buffer", "Waiting for writing "
                    "thread to finish.");
            m_writing_thread.join();
        }
        m_write_flag = true;
        tmd::debug("WriteBuffer", "flush_buffer", "Preparing the write "
                "thread.");
        std::vector<frame_t*> buffer_cpy = m_buffer;
        m_buffer.clear();
        m_writing_thread = std::thread(_write_buffer_to_disk, buffer_cpy,
                                  m_dest_folder,
                                  m_save_mask, &m_elements_count,
                                  std::ref(m_write_flag));
    }
}