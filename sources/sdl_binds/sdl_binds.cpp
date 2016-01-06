#include "../../headers/sdl_binds/sdl_binds.h"

namespace tmd{
    bool SDLBinds::ms_sdl_init = false;

    void SDLBinds::init_sdl(){
        SDL_Init(SDL_INIT_VIDEO);
    }

    SDL_Surface* SDLBinds::get_sdl_surface_from_mat(cv::Mat mat){
        IplImage opencvimg2 = (IplImage)mat;
        IplImage* opencvimg = &opencvimg2;

        //Convert to SDL_Surface
        SDL_Surface* frameSurface = SDL_CreateRGBSurfaceFrom((void*)
                                                     opencvimg->imageData,
                                        opencvimg->width, opencvimg->height,
                                        opencvimg->depth*opencvimg->nChannels,
                                        opencvimg->widthStep,
                                        0xff0000, 0x00ff00, 0x0000ff, 0);

        if(frameSurface == NULL) {
            tmd::debug("SDLBinds", "get_sdl_surface_from_mat", "Couldn't "
                    "create surface from Mat.");
        }
        return frameSurface;
    }

    SDL_Window* SDLBinds::create_sdl_window(std::string name, int w, int h){
        if (!ms_sdl_init){
            SDL_Init(SDL_INIT_VIDEO);
            ms_sdl_init = true;
        }

        SDL_Window *window;                    // Declare a pointer

        // Create an application window with the following settings:
        window = SDL_CreateWindow(
                name.c_str(),                  // window title
                SDL_WINDOWPOS_UNDEFINED,           // initial x position
                SDL_WINDOWPOS_UNDEFINED,           // initial y position
                w,                               // width, in pixels
                h,                               // height, in pixels
                SDL_WINDOW_RESIZABLE                  // flags
        );

        // Check that the window was successfully created
        if (window == NULL) {
            // In the case that the window could not be made...
            tmd::debug("SDLBinds", "create_sdl_window", "Could not create "
                    "window: " + std::string(SDL_GetError()));
        }
        return window;
    }

    void SDLBinds::imshow(SDL_Window* window, cv::Mat frame){
        SDL_SetWindowSize(window, frame.cols, frame.rows);
        SDL_Surface* screen = SDL_GetWindowSurface(window);
        SDL_Surface* image = get_sdl_surface_from_mat(frame);
        SDL_BlitSurface(image, NULL, screen, NULL); // blit it to the screen
        SDL_FreeSurface(image);
        SDL_UpdateWindowSurface(window);
    }

    void SDLBinds::destroy_sdl_window(SDL_Window* window){
        SDL_DestroyWindow(window);
    }

    void SDLBinds::quit_sdl(){
        SDL_Quit();
    }
}