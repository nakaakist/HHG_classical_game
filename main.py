'''
HHG GAME
'''
import numpy as np

import pyglet
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

WINDOW_WIDTH  = 1000
WINDOW_HEIGHT = 700

ELECTRON_DRAW_PARAMS = {'diffuse': [0, 1, 0, 1],
                        'specular': [1, 1, 1, 1],
                        'shininess': 40,
                        'slices': 10,
                        'stacks': 5}

NUCLEAR_DRAW_PARAMS = {'diffuse': [1, 0, 0, 1],
                       'specular': [1, 1, 1, 1],
                       'shininess': 40,
                       'slices': 15,
                       'stacks': 15}

ATOM_RADIUS = 5
ELECTRON_INITIAL_POSITION = [ATOM_RADIUS, 0, 0]
ELECTRON_INITIAL_VELOCITY = [0, 1/np.sqrt(ATOM_RADIUS), 0]
ELECTRON_ANGULAR_FREQUENCY = 1/np.sqrt(ATOM_RADIUS)**3

TIME_SCALE_FACTOR = 100

FONT_NAME = ('Verdana', 'Helvetica', 'Arial')

INSTRUCTIONS = \
'''Your ship is lost in a peculiar unchartered area of space-time infested with asteroids!  You have no chance for survival except to rack up the highest score possible.

Left/Right: Turn ship
Up: Thrusters
Space: Shoot

Be careful, there's not much friction in space.'''

# --------------------------------------------------------------------------
# Game objects
# --------------------------------------------------------------------------

class Particle:
  def __init__(self, size, position, draw_params):
    #color_params is a dictionary like {'diffuse': [1, 0, 0, 1], 'ambient': ...}
    self.size = size
    self.x, self.y, self.z = position
    self.draw_params = draw_params

  def draw(self):
    glPushMatrix()
    glTranslated(self.x, self.y, self.z)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self.draw_params['diffuse'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.draw_params['specular'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, self.draw_params['shininess'])
    glutSolidSphere(self.size, self.draw_params['slices'], self.draw_params['stacks'])
    glPopMatrix()

class Nuclear(Particle):
  def __init__(self, size, position, draw_params):
    super().__init__(size, position, draw_params)

  def update(self):
    pass

class Electron_ionized(Particle):
  def __init__(self, size, position, velocity, draw_params):
    super().__init__(size, position, draw_params)
    self.vx, self.vy, self.vz = velocity

  def update(self, dt, e):
    self.__RK4_xy(dt, e[0], e[1])

  def __coulomb_force_xy(self, x, y):
    return -x/np.sqrt(x**2+y**2)**3

  def __RK4_xy(self, dt, ex, ey):
    kx1 = dt*self.vx
    kvx1 = dt*(self.__coulomb_force_xy(self.x, self.y)+ex)
    ky1 = dt*self.vy
    kvy1 = dt*(self.__coulomb_force_xy(self.y, self.x)+ey)
    kx2 = dt*(self.vx+kvx1/2)
    kvx2 = dt*(self.__coulomb_force_xy(self.x+kx1/2, self.y+ky1/2)+ex)
    ky2 = dt*(self.vy+kvy1/2)
    kvy2 = dt*(self.__coulomb_force_xy(self.y+ky1/2, self.x+kx1/2)+ey)
    kx3 = dt*(self.vx+kvx2/2)
    kvx3 = dt*(self.__coulomb_force_xy(self.x+kx2/2, self.y+ky2/2)+ex)
    ky3 = dt*(self.vy+kvy2/2)
    kvy3 = dt*(self.__coulomb_force_xy(self.y+ky2/2, self.x+kx2/2)+ey)
    kx4 = dt*(self.vx+kvx3)
    kvx4 = dt*(self.__coulomb_force_xy(self.x+kx3, self.y+ky3)+ex)
    ky4 = dt*(self.vy+kvy3)
    kvy4 = dt*(self.__coulomb_force_xy(self.y+ky3, self.x+kx3)+ey)
    self.x += (kx1+2*kx2+2*kx3+kx4)/6
    self.y += (ky1+2*ky2+2*ky3+ky4)/6
    self.vx += (kvx1+2*kvx2+2*kvx3+kvx4)/6
    self.vy += (kvy1+2*kvy2+2*kvy3+kvy4)/6

class Electron_localized(Particle):
  def __init__(self, size, position, velocity, draw_params):
    super().__init__(size, position, draw_params)
    self.vx, self.vy, self.vz = velocity
    if self.x == 0:
      self.angle0 = np.pi/2 if self.y > 0 else -np.pi/2
    else:
      self.angle0 = np.arctan(self.y/self.x)
    self.r = np.sqrt(self.x**2+self.y**2)
    self.t = 0

  def update(self, dt):
    self.x = self.r*np.cos(ELECTRON_ANGULAR_FREQUENCY*self.t+self.angle0)
    self.y = self.r*np.sin(ELECTRON_ANGULAR_FREQUENCY*self.t+self.angle0)
    self.t += dt

# --------------------------------------------------------------------------
# Overlays, such as menus and "Game Over" banners
# --------------------------------------------------------------------------

class Overlay(object):
    def update(self, dt):
        pass

    def draw(self):
        pass

class Banner(Overlay):
    def __init__(self, label, dismiss_func=None, timeout=None):
        self.text = pyglet.text.Label(label,
                                      font_name=FONT_NAME,
                                      font_size=36,
                                      x=ARENA_WIDTH // 2,
                                      y=ARENA_HEIGHT // 2,
                                      anchor_x='center',
                                      anchor_y='center')

        self.dismiss_func = dismiss_func
        self.timeout = timeout
        if timeout and dismiss_func:
            pyglet.clock.schedule_once(dismiss_func, timeout)

    def draw(self):
        self.text.draw()

    def on_key_press(self, symbol, modifiers):
        if self.dismiss_func and not self.timeout:
            self.dismiss_func()
        return True

class Menu(Overlay):
    def __init__(self, title):
        self.items = []
        self.title_text = pyglet.text.Label(title, 
                                            font_name=FONT_NAME,
                                            font_size=36,
                                            x=ARENA_WIDTH // 2, 
                                            y=350,
                                            anchor_x='center',
                                            anchor_y='center')

    def reset(self):
        self.selected_index = 0
        self.items[self.selected_index].selected = True

    def on_key_press(self, symbol, modifiers):
        if symbol == key.DOWN:
            self.selected_index += 1
        elif symbol == key.UP:
            self.selected_index -= 1
        self.selected_index = min(max(self.selected_index, 0), 
                                  len(self.items) - 1)

        if symbol in (key.DOWN, key.UP) and enable_sound:
            bullet_sound.play()

    def on_key_release(self, symbol, modifiers):
        self.items[self.selected_index].on_key_release(symbol, modifiers)

    def draw(self):
        self.title_text.draw()
        for i, item in enumerate(self.items):
            item.draw(i == self.selected_index)

class MenuItem(object):
    pointer_color = (.46, 0, 1.)
    inverted_pointers = False

    def __init__(self, label, y, activate_func):
        self.y = y
        self.text = pyglet.text.Label(label,
                                      font_name=FONT_NAME,
                                      font_size=14,
                                      x=ARENA_WIDTH // 2, 
                                      y=y,
                                      anchor_x='center',
                                      anchor_y='center')
        self.activate_func = activate_func

    def draw_pointer(self, x, y, color, flip=False):
        # Tint the pointer image to a color
        glPushAttrib(GL_CURRENT_BIT)
        glColor3f(*color)
        if flip:
            pointer_image_flip.blit(x, y)
        else:
            pointer_image.blit(x, y)
        glPopAttrib()

    def draw(self, selected):
        self.text.draw()

        if selected:
            self.draw_pointer(
                self.text.x - self.text.content_width // 2 - 
                    pointer_image.width // 2,
                self.y, 
                self.pointer_color,
                self.inverted_pointers)
            self.draw_pointer(
                self.text.x + self.text.content_width // 2 + 
                    pointer_image.width // 2,
                self.y,
                self.pointer_color,
                not self.inverted_pointers)

    def on_key_release(self, symbol, modifiers):
        if symbol == key.ENTER and self.activate_func:
            self.activate_func()
            if enable_sound:
                bullet_sound.play()

class ToggleMenuItem(MenuItem):
    pointer_color = (.27, .82, .25)
    inverted_pointers = True

    def __init__(self, label, value, y, toggle_func):
        self.value = value
        self.label = label
        self.toggle_func = toggle_func
        super(ToggleMenuItem, self).__init__(self.get_label(), y, None)

    def get_label(self):
        return self.label + (self.value and ': ON' or ': OFF')

    def on_key_release(self, symbol, modifiers):
        if symbol == key.LEFT or symbol == key.RIGHT:
            self.value = not self.value
            self.text.text = self.get_label()
            self.toggle_func(self.value)
            if enable_sound:
                bullet_sound.play()

class DifficultyMenuItem(MenuItem):
    pointer_color = (.27, .82, .25)
    inverted_pointers = True

    def __init__(self, y):
        super(DifficultyMenuItem, self).__init__(self.get_label(), y, None)

    def get_label(self):
        if difficulty == 0:
            return 'Difficulty: Pebbles'
        elif difficulty == 1:
            return 'Difficulty: Stones'
        elif difficulty == 2:
            return 'Difficulty: Asteroids'
        elif difficulty == 3:
            return 'Difficulty: Meteors'
        else:
            return 'Difficulty: %d' % difficulty

    def on_key_release(self, symbol, modifiers):
        global difficulty
        if symbol == key.LEFT:
            difficulty -= 1
        elif symbol == key.RIGHT:
            difficulty += 1
        difficulty = min(max(difficulty, 0), MAX_DIFFICULTY)
        self.text.text = self.get_label()

        if symbol in (key.LEFT, key.RIGHT) and enable_sound:
            bullet_sound.play()

class MainMenu(Menu):
    def __init__(self):
        super(MainMenu, self).__init__('Astraea')

        self.items.append(MenuItem('New Game', 240, begin_game))
        self.items.append(MenuItem('Instructions', 200, 
                                   begin_instructions_menu))
        self.items.append(MenuItem('Options', 160, begin_options_menu))
        self.items.append(MenuItem('Quit', 120, sys.exit))
        self.reset()

class OptionsMenu(Menu):
    def __init__(self):
        super(OptionsMenu, self).__init__('Options')

        self.items.append(DifficultyMenuItem(280))
        def set_enable_sound(value):
            global enable_sound
            enable_sound = value
        self.items.append(ToggleMenuItem('Sound', enable_sound, 240,
                                         set_enable_sound))

        def set_enable_fullscreen(value):
            win.set_fullscreen(value, width=ARENA_WIDTH, height=ARENA_HEIGHT)
        self.items.append(ToggleMenuItem('Fullscreen', win.fullscreen, 200,
                                         set_enable_fullscreen))
                                
        self.items.append(ToggleMenuItem('Vsync', win.vsync, 160, 
                                         win.set_vsync))

        def set_show_fps(value):
            global show_fps
            show_fps = value
        self.items.append(ToggleMenuItem('FPS', show_fps, 120, set_show_fps))
        self.items.append(MenuItem('Ok', 60, begin_main_menu))
        self.reset()

class InstructionsMenu(Menu):
    def __init__(self):
        super(InstructionsMenu, self).__init__('Instructions')

        self.items.append(MenuItem('Ok', 50, begin_main_menu))
        self.reset()

        self.instruction_text = pyglet.text.Label(INSTRUCTIONS,
                                                  font_name=FONT_NAME,
                                                  font_size=14,
                                                  x=20, y=300,
                                                  width=ARENA_WIDTH - 40,
                                                  anchor_y='top',
                                                  multiline=True)

    def draw(self):
        super(InstructionsMenu, self).draw()
        self.instruction_text.draw()

class PauseMenu(Menu):
    def __init__(self):
        super(PauseMenu, self).__init__('Paused')

        self.items.append(MenuItem('Continue Game', 240, resume_game))
        self.items.append(MenuItem('Main Menu', 200, end_game))
        self.reset()

# --------------------------------------------------------------------------
# Game state functions
# --------------------------------------------------------------------------

def check_collisions():
    # Check for collisions using an approximate uniform grid.
    #
    #   1. Mark all grid cells that the bullets are in
    #   2. Mark all grid cells that the player is in
    #   3. For each asteroid, check grid cells that are covered for
    #      a collision.
    #
    # This is by no means perfect collision detection (in particular,
    # there are rounding errors, and it doesn't take into account the
    # arena wrapping).  Improving it is left as an exercise for the
    # reader.

    # The collision grid.  It is recreated each iteration, as bullets move
    # quickly.
    hit_squares = {}

    # 1. Mark all grid cells that the bullets are in.  Assume bullets
    #    occupy a single cell.
    for bullet in bullets:
        hit_squares[int(bullet.x / COLLISION_RESOLUTION), 
                    int(bullet.y / COLLISION_RESOLUTION)] = bullet

    # 2. Mark all grid cells that the player is in.
    for x, y in player.collision_cells():
        hit_squares[x, y] = player

    # 3. Check grid cells of each asteroid for a collision.
    for asteroid in asteroids:
        for x, y in asteroid.collision_cells():
           if (x, y) in hit_squares:
                asteroid.hit = True
                hit_squares[x, y].hit = True
                del hit_squares[x, y]

def begin_main_menu():
    set_overlay(MainMenu())

def begin_options_menu():
    set_overlay(OptionsMenu())

def begin_instructions_menu():
    set_overlay(InstructionsMenu())

def begin_game():
    global player_lives
    global score
    player_lives = 3
    score = 0

    begin_clear_background()
    set_overlay(Banner('Get Ready', begin_first_round, GET_READY_DELAY))

def begin_first_round(*args):
    player.reset()
    player.visible = True
    begin_round()

def next_round(*args):
    global in_game
    player.invincible = True
    in_game = False
    set_overlay(Banner('Get Ready', begin_round, GET_READY_DELAY))

def begin_round(*args):
    global asteroids
    global bullets
    global animations
    global in_game
    asteroids = []
    for i in range(INITIAL_ASTEROIDS[difficulty]):
        x = random.random() * ARENA_WIDTH
        y = random.random() * ARENA_HEIGHT
        asteroids.append(Asteroid(asteroid_sizes[-1], x, y, wrapping_batch))

    for bullet in bullets:
        bullet.delete()

    for animation in animations:
        animation.delete()

    bullets = []
    animations = []
    in_game = True
    set_overlay(None)
    pyglet.clock.schedule_once(begin_play, BEGIN_PLAY_DELAY)

def begin_play(*args):
    player.invincible = False

def begin_life(*args):
    player.reset()
    pyglet.clock.schedule_once(begin_play, BEGIN_PLAY_DELAY)

def life_lost(*args):
    global player_lives
    player_lives -= 1

    if player_lives > 0:
        begin_life()
    else:
        game_over()

def game_over():
    set_overlay(Banner('Game Over', end_game))

def pause_game():
    global paused
    paused = True
    set_overlay(PauseMenu())

def resume_game():
    global paused
    paused = False
    set_overlay(None)

def end_game():
    global in_game
    global paused
    paused = False
    in_game = False
    player.invincible = True
    pyglet.clock.unschedule(life_lost)
    pyglet.clock.unschedule(begin_play)
    begin_menu_background()
    set_overlay(MainMenu())

def set_overlay(new_overlay):
    global overlay
    if overlay:
        win.remove_handlers(overlay)
    overlay = new_overlay
    if overlay:
        win.push_handlers(overlay)

def begin_menu_background():
    global asteroids
    global bullets
    global animations
    global in_game
    global player_lives

    asteroids = []
    for i in range(11):
        x = random.random() * ARENA_WIDTH
        y = random.random() * ARENA_HEIGHT
        asteroids.append(Asteroid(asteroid_sizes[i // 4], x, y, wrapping_batch))

    for bullet in bullets:
        bullet.delete()

    for animation in animations:
        animation.delete()

    bullets = []
    animations = []
    in_game = False
    player_lives = 0
    player.visible = False

def begin_clear_background():
    global asteroids
    global bullets
    global animations

    for bullet in bullets:
        bullet.delete()

    for animation in animations:
        animation.delete()

    asteroids = []
    bullets = []
    animations = []
    player.visible = False

# --------------------------------------------------------------------------
# Create window
# --------------------------------------------------------------------------

win = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption='HHG Game')

@win.event
def on_draw():
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()
  gluLookAt(0, 0, 50, 0.0, 0.0, 0.0, 0.0, 1, 0)
  glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 0.0])
  electron_ionized.draw()
  electron_localized.draw()
  nuclear.draw()

@win.event
def on_resize(width, height):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glViewport(0, 0, width, height)
  gluPerspective(45, width//height, 0.1, 1000)
  glMatrixMode(GL_MODELVIEW)
  return pyglet.event.EVENT_HANDLED

# --------------------------------------------------------------------------
# Global game state vars
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Game update
# --------------------------------------------------------------------------

def update(dt):
  dt_scaled = dt*TIME_SCALE_FACTOR
  electron_ionized.update(dt_scaled, [0, 0])
  electron_localized.update(dt_scaled)
  nuclear.update()

pyglet.clock.schedule_interval(update, 1/60.)

# --------------------------------------------------------------------------
# Start game
# --------------------------------------------------------------------------
def init_gl():
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
  glEnable(GL_CULL_FACE)
  glCullFace(GL_BACK)
  glEnable(GL_DEPTH_TEST)
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
  glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

electron_ionized = Electron_ionized(3, [10, 0, 0], [0, 1, 0], ELECTRON_DRAW_PARAMS)
electron_localized = Electron_localized(3, [10, 0, 0], [0, 1, 0], ELECTRON_DRAW_PARAMS)
nuclear = Nuclear(5, [0, 0, 0], NUCLEAR_DRAW_PARAMS)

init_gl()
pyglet.app.run()

