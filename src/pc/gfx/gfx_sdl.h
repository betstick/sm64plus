#ifndef GFX_SDL_H
#define GFX_SDL_H

#include "gfx_window_manager_api.h"

extern struct GfxWindowManagerAPI gfx_sdl;

extern struct SDL_Window* get_sdl_window();

extern struct SDL_GLContext* get_sdl_gl_context();

bool poll_events(struct SDL_Event** event);

bool should_close(struct SDL_Event** event);

#endif
