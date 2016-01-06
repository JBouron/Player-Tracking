#ifndef BACHELOR_PROJECT_SDL_BINDS_H
#define BACHELOR_PROJECT_SDL_BINDS_H

#include <SDL_surface.h>
#include <opencv2/core/core.hpp>
#include <SDL_video.h>
#include <SDL.h>
#include "../misc/config.h"

namespace tmd{
    /**
     * Static class for operations involving SDL.
     */
    class SDLBinds{
    public:
        /**
         * Init the SDL. Called automatically if the user is trying to create
         * a window before calling init SDL.
         */
        static void init_sdl();

        /**
         * Allow us to convert an image matrix into an SDL_surface*.
         */
        static SDL_Surface* get_sdl_surface_from_mat(cv::Mat mat);

        static SDL_Window* create_sdl_window(std::string name,
                                     int w = Config::sdl_binds_default_width,
                                     int h = Config::sdl_binds_default_height);

        static void imshow(SDL_Window* window, cv::Mat frame);

        static void destroy_sdl_window(SDL_Window* window);

        static void quit_sdl();

        static bool ms_sdl_init;
    };
}

#endif //BACHELOR_PROJECT_SDL_BINDS_H
