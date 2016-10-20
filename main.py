'''
HHG GAME
'''
import numpy as np
import pandas as pd
import os

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
                       'slices': 20,
                       'stacks': 20}

ATOM_RADIUS = 5
IONIZATION_RADIUS = 7
ELECTRON_SIZE = 1
NUCLEAR_SIZE = 1.5
ELECTRON_INITIAL_POSITION = [ATOM_RADIUS, 0, 0]
ELECTRON_INITIAL_VELOCITY = [0, 1/np.sqrt(ATOM_RADIUS), 0]
ELECTRON_ANGULAR_FREQUENCY = 1/np.sqrt(ATOM_RADIUS)**3
ELECTRON_CACHE_NUM = 30
LIGHT_CONE_SLICES = 50
LIGHT_CONE_STACKS = 10
LIGHT_CONE_BOTTOM = 20
LIGHT_CONE_HEIGHT = 150
LIGHT_CONE_COLOR_MAX_ENERGY = 40
LIGHT_FLASH_MAX_ENERGY = 80
X_MAX = 60
Y_MAX = 60
MESH_INTERVAL = 5
RYDBERG = 27.21

TIME_SCALE_FACTOR = 20
ELECTRIC_FIELD_SCALE_FACTOR = 0.0001

ELECTRIC_FIELD_WINDOW_FRACTION = 0.1

FONT_NAME = 'Osaka'
RANKING_FILENAME = 'tmp/ranking.csv'

MENU_TITLE_SIZE = 36
MENU_TITLE_POSITION = WINDOW_HEIGHT/8
MENU_ITEM_SIZE = 24
MENU_ITEM_INTERVAL = WINDOW_HEIGHT/15

START_GAME_DELAY = 1
CLEAR_DELAY = 2

VIEW_START_X = 0
VIEW_START_Y = 50
VIEW_START_Z = 50

VIEW_GAME_X = 0
VIEW_GAME_Y = 0
VIEW_GAME_Z = 150


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
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.draw_params['diffuse'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.draw_params['specular'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, self.draw_params['shininess'])
    glutSolidSphere(self.size, self.draw_params['slices'], self.draw_params['stacks'])
    glPopMatrix()

  def update(self):
    pass

class Nuclear(Particle):
  def __init__(self, size, position, draw_params):
    super().__init__(size, position, draw_params)

class Electron_ionized(Particle):
  def __init__(self, size, position, velocity, draw_params):
    super().__init__(size, position, draw_params)
    self.vx, self.vy, self.vz = velocity
    self.position_cache = [position]*ELECTRON_CACHE_NUM
    self.is_active = True

  def update(self, dt, e):
    self.__RK4_xy(dt, -e[0], -e[1])
    self.position_cache = self.position_cache[1:]+[[self.x, self.y, self.z]]

  def draw(self):
    if self.is_active:
      for i, p in enumerate(self.position_cache):
        if i%2 == 0:
          attenuation = np.exp(-(1-(i+1)/len(self.position_cache)))
          glPushMatrix()
          glTranslated(p[0], p[1], p[2])
          diffuse = self.draw_params['diffuse']
          diffuse[3] = attenuation**2
          glEnable(GL_BLEND)
          glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.draw_params['diffuse'])
          glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.draw_params['specular'])
          glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, self.draw_params['shininess'])
          if i == len(self.position_cache)-2:
            glDisable(GL_BLEND)
          glutSolidSphere(self.size*attenuation, self.draw_params['slices'], int(self.draw_params['stacks']*attenuation))
          glPopMatrix()

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

class Electric_field:
  def __init__(self):
    self.ex = 0
    self.ey = 0

  def draw(self):
    glColor3d(1.0, 0.0, 0.0)
    glBegin(GL_POLYGON)
    glVertex2d(-WINDOW_WIDTH/2, -WINDOW_HEIGHT/2)
    glVertex2d(-WINDOW_WIDTH/2, -WINDOW_HEIGHT*(0.5-ELECTRIC_FIELD_WINDOW_FRACTION))
    glVertex2d(-WINDOW_WIDTH*(0.5-ELECTRIC_FIELD_WINDOW_FRACTION),
               -WINDOW_HEIGHT*(0.5-ELECTRIC_FIELD_WINDOW_FRACTION))
    glVertex2d(-WINDOW_WIDTH*(0.5-ELECTRIC_FIELD_WINDOW_FRACTION), -WINDOW_HEIGHT/2)
    glEnd()

class LightCone:
  def __init__(self):
    self.is_active = False
    self.angle = 0

  def draw(self):
    if self.is_active:
      glEnable(GL_BLEND)
      glEnable(GL_LIGHT1)
      glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
      glLightfv(GL_LIGHT1, GL_SPECULAR, [50.0, 50.0, 50.0, 10.0])
      glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, 5.0, 1.0])
      for axis in [-1, 1]:
        glPushMatrix()
        glRotated(self.angle, 0, 0, 1)
        glRotated(90, 0, axis, 0)
        glTranslated(0, 0, -LIGHT_CONE_HEIGHT)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.__mix_color())
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1, 1, 1, 1])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glutSolidCone(LIGHT_CONE_BOTTOM,
                      LIGHT_CONE_HEIGHT,
                      LIGHT_CONE_SLICES,
                      LIGHT_CONE_STACKS)
        glPopMatrix()
      glDisable(GL_LIGHT1)
      glDisable(GL_BLEND)

  def __mix_color(self):
    global score
    if score < 0.3*LIGHT_CONE_COLOR_MAX_ENERGY:
      return [0, 0, (score/0.3/LIGHT_CONE_COLOR_MAX_ENERGY), 0.4]
    elif score < 0.8*LIGHT_CONE_COLOR_MAX_ENERGY:
      return [(score/0.5/LIGHT_CONE_COLOR_MAX_ENERGY)-0.3/0.5, 0, 1, 0.4]
    elif score < LIGHT_CONE_COLOR_MAX_ENERGY:
      return [1, (score/0.5/LIGHT_CONE_COLOR_MAX_ENERGY)-0.8/0.5, 1, 0.4]
    else:
      return [1, 0.4, 1, 0.4]

class Mesh:
  def __init__(self, xmax, ymax, mesh_interval):
    self.xmax = xmax
    self.ymax = ymax
    self.mesh_interval = mesh_interval

  def draw(self):
    glBegin(GL_LINES)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [1, 1, 1, 1])
    for x in np.arange(-self.xmax, self.xmax+self.mesh_interval, self.mesh_interval):
      glVertex3d(x, -self.xmax, 0)
      glVertex3d(x, self.xmax, 0)
    for y in np.arange(-self.xmax, self.xmax+self.mesh_interval, self.mesh_interval):
      glVertex3d(-self.xmax, y, 0)
      glVertex3d(self.xmax, y, 0)
    glEnd()

# --------------------------------------------------------------------------
# Overlays, such as menus and "Game Over" banners
# --------------------------------------------------------------------------

class Overlay:
  def update(self, dt):
    pass

  def draw(self):
    pass

class Banner(Overlay):
  def __init__(self, label, invoke_func):
    self.invoke_func = invoke_func
    self.text = pyglet.text.Label(label,
                                  font_name=FONT_NAME,
                                  font_size=36,
                                  x=0,
                                  y=MENU_TITLE_POSITION,
                                  anchor_x='center',
                                  anchor_y='center')
  def draw(self):
    self.text.draw()

  def on_mouse_press(self, x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
      self.invoke_func()

class TextList(Overlay):
  def __init__(self, title, invoke_func):
    self.items = []
    self.invoke_func = invoke_func
    self.title_text = pyglet.text.Label(title,
                                        font_name=FONT_NAME,
                                        font_size=MENU_TITLE_SIZE,
                                        x=0,
                                        y=MENU_TITLE_POSITION,
                                        anchor_x='center',
                                        anchor_y='center')

  def append_label(self, label):
    item = TextListItem(label, -(len(self.items)+1)*MENU_ITEM_INTERVAL)
    self.items.append(item)

  def draw(self):
    self.title_text.draw()
    [item.draw() for item in self.items]

  def on_mouse_press(self, x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
      self.invoke_func()

class TextListItem(Overlay):
  def __init__(self, label, y):
    self.text = pyglet.text.Label(label,
                                  font_name=FONT_NAME,
                                  font_size=MENU_ITEM_SIZE,
                                  x=0,
                                  y=y,
                                  anchor_x='center',
                                  anchor_y='center')

  def draw(self):
    self.text.draw()

class GameOver(TextList):
  def __init__(self):
    super().__init__('電子を見失った!', show_start_menu)
    self.append_label('左clickでメニューへ')

class Cleared(TextList):
  def __init__(self, score):
    super().__init__('高調波発生!', show_ranking_after_clear)
    self.append_label('エネルギー %d eV' % score)
    self.append_label('左clickで次へ' % score)

class Ranking(TextList):
  def __init__(self):
    super().__init__('ランキング', show_start_menu)
    if os.path.exists(RANKING_FILENAME):
      ranking_data = pd.read_csv(RANKING_FILENAME).sort_values(by='score', ascending=False)
      for i in range(min(5, ranking_data.index.size)):
        self.append_label('%d位: %s さん  %d eV' % (i+1, ranking_data.iloc[i]['name'], ranking_data.iloc[i]['score']))
    else:
      self.append_label('データなし')
    self.append_label('左clickで戻る')

class RankingAfterClear(TextList):
  def __init__(self):
    global score, ranking_data
    if ranking_data is None:
      super().__init__('ハイスコア! あなたの順位は1位です', input_name_for_ranking)
      rank_text = TextListItem('1位: あなた  %d eV' % score, -MENU_ITEM_INTERVAL)
      rank_text.text.color = [255]*4
      self.items.append(rank_text)
    else:
      rank = ranking_data[ranking_data.score > score].index.size+1
      if rank < 5:
        super().__init__('ハイスコア! あなたの順位は%d位です' % rank, input_name_for_ranking)
        i_rank = 0
        for i in range(min(5, ranking_data.index.size)):
          if i_rank == rank-1:
            rank_text = TextListItem('%d位: あなた  %d eV' % (i_rank+1, score), -MENU_ITEM_INTERVAL*(i_rank+1))
            rank_text.text.color = [255, 255, 0, 255]
            self.items.append(rank_text)
          else:
            self.append_label('%d位: %s さん  %d eV' % (i_rank+1, ranking_data.iloc[i_rank]['name'], ranking_data.iloc[i_rank]['score']))
          i_rank += 1
      else:
        super().__init__('ランク外。あなたの順位は%d位です' % rank, show_start_menu)
        for i in range(min(5, ranking_data.index.size)):
          self.append_label('%d位: %s さん  %d eV' % (i+1, ranking_data.iloc[i]['name'], ranking_data.iloc[i]['score']))
    self.append_label('左clickで次へ')

class InputName(Overlay):
  def __init__(self):
    self.name = ''
    self.title_text = pyglet.text.Label('名前を入力',
                                        font_name=FONT_NAME,
                                        font_size=MENU_TITLE_SIZE,
                                        x=0,
                                        y=MENU_TITLE_POSITION,
                                        anchor_x='center',
                                        anchor_y='center')

  def on_text(self, text):
    self.name += text

  def on_key_press(self, symbol, modifiers):
    if symbol == pyglet.window.key.ENTER:
      global ranking_data, score
      self.name = self.name.rstrip()
      current_data = pd.DataFrame([self.name, score]).T.rename(columns={0: 'name', 1: 'score'})
      if ranking_data is not None:
        appended_ranking_data = ranking_data.append(current_data)
      else:
        appended_ranking_data = current_data
      appended_ranking_data.to_csv(RANKING_FILENAME, index=False)
      show_start_menu()

  def draw(self):
    self.title_text.draw()
    if self.name:
      self.text = pyglet.text.Label(self.name,
                                    font_name=FONT_NAME,
                                    font_size=MENU_ITEM_SIZE,
                                    x=0,
                                    y=-100,
                                    anchor_x='center',
                                    anchor_y='center')
      self.text.draw()

class Menu(Overlay):
  def __init__(self, title):
    self.items = []
    self.caption = None
    self.selected_index = 0
    self.title_text = pyglet.text.Label(title,
                                        font_name=FONT_NAME,
                                        font_size=MENU_TITLE_SIZE,
                                        x=0,
                                        y=MENU_TITLE_POSITION,
                                        anchor_x='center',
                                        anchor_y='center')

  def on_mouse_press(self, x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
      i = -int((y-WINDOW_HEIGHT//2+0.5*MENU_ITEM_INTERVAL)/MENU_ITEM_INTERVAL)
      if (i >= 0 ) and (i < len(self.items)):
        self.items[i].invoke_func()

  def add_caption(self, label):
    self.caption = pyglet.text.Label(label,
                                     font_name=FONT_NAME,
                                     font_size=MENU_ITEM_SIZE,
                                     x=0,
                                     y=-MENU_ITEM_INTERVAL*(len(self.items)+1),
                                     anchor_x='center',
                                     anchor_y='center')

  def draw(self):
    self.title_text.draw()
    for i, item in enumerate(self.items):
      item.draw()
    self.caption.draw()

class MenuItem(object):
  def __init__(self, label, y, invoke_func):
    self.y = y
    self.invoke_func = invoke_func
    self.text = pyglet.text.Label(label,
                                  font_name=FONT_NAME,
                                  font_size=MENU_ITEM_SIZE,
                                  x=0,
                                  y=y,
                                  anchor_x='center',
                                  anchor_y='center')

  def draw(self):
    self.text.draw()

class StartMenu(Menu):
  def __init__(self):
    super().__init__('高次高調波発生ゲーム')
    self.items.append(MenuItem('遊ぶ', -MENU_ITEM_INTERVAL, start_game_transition))
    self.items.append(MenuItem('ランキング', -MENU_ITEM_INTERVAL*2, show_ranking))
    self.items.append(MenuItem('遊び方', -MENU_ITEM_INTERVAL*3, start_tutorial_transition))
    self.add_caption('左clickで選択')

# --------------------------------------------------------------------------
# In game event handler
# --------------------------------------------------------------------------

class InGameEventHandler(object):
  def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
    global electric_field
    if buttons & pyglet.window.mouse.LEFT:
      electric_field.ex = ELECTRIC_FIELD_SCALE_FACTOR*(x-WINDOW_WIDTH/2)
      electric_field.ey = ELECTRIC_FIELD_SCALE_FACTOR*(y-WINDOW_HEIGHT/2)

  def on_mouse_release(self, x, y, button, modifiers):
    global electric_field
    electric_field.ex = 0
    electric_field.ey = 0

# --------------------------------------------------------------------------
# Game state functions
# --------------------------------------------------------------------------

def init_objects():
  global electron_ionized, electron_localized, nuclear, electric_field, light_cone, mesh
  electron_ionized = Electron_ionized(ELECTRON_SIZE, ELECTRON_INITIAL_POSITION, ELECTRON_INITIAL_VELOCITY, ELECTRON_DRAW_PARAMS)
  electron_localized = Electron_localized(ELECTRON_SIZE, ELECTRON_INITIAL_POSITION, ELECTRON_INITIAL_VELOCITY, ELECTRON_DRAW_PARAMS)
  nuclear = Nuclear(NUCLEAR_SIZE, [0, 0, 0], NUCLEAR_DRAW_PARAMS)
  electric_field = Electric_field()
  light_cone = LightCone()
  mesh = Mesh(X_MAX, Y_MAX, MESH_INTERVAL)

def set_overlay(new_overlay):
  global overlay
  if overlay:
    win.remove_handlers(overlay)
  overlay = new_overlay
  if overlay:
    win.push_handlers(overlay)

def start_game_transition():
  global in_start_menu, in_transition_from_start_menu_to_game, t_transition
  set_overlay(None)
  in_start_menu = False
  in_transition_from_start_menu_to_game = True
  t_transition = 0
  pyglet.clock.schedule_once(start_game, START_GAME_DELAY)

def start_tutorial_transition():
  in_tutorial = True
  start_game_transition()


def start_game(dt):
  global in_transition_from_start_menu_to_game, in_game_event_handler, in_game
  in_game_event_handler = InGameEventHandler()
  set_overlay(None)
  win.push_handlers(in_game_event_handler)
  in_transition_from_start_menu_to_game = False
  in_game = True

def show_ranking():
 set_overlay(Ranking())

def check_gameover():
  global electron_ionized, in_game, is_gameover, is_ionized
  if np.abs(electron_ionized.x) > X_MAX or np.abs(electron_ionized.y) > Y_MAX:
    in_game = False
    is_ionized = False
    is_gameover = True
    win.remove_handlers(in_game_event_handler)
    set_overlay(GameOver())

def check_collision():
  global electron_ionized, in_game, is_cleared, is_ionized, score, light_cone
  r = np.sqrt(electron_ionized.x**2+electron_ionized.y**2)
  if (not is_ionized) and (r > IONIZATION_RADIUS) and (not is_gameover):
    is_ionized = True
  if is_ionized and r < ATOM_RADIUS:
    energy = np.sqrt(electron_ionized.vx**2+electron_ionized.vy**2)/2
    score = energy*RYDBERG
    in_game = False
    is_ionized = False
    electron_ionized.is_active = False
    light_cone.is_active = True
    light_cone.angle = 180/np.pi*np.arctan(-electron_ionized.vx/electron_ionized.vy)
    win.remove_handlers(in_game_event_handler)
    clear_transition()

def clear_transition():
  global in_transition_from_game_to_cleared, t_transition
  set_overlay(None)
  in_transition_from_game_to_cleared = True
  t_transition = 0
  pyglet.clock.schedule_once(cleared, CLEAR_DELAY)

def cleared(dt):
  global in_transition_from_game_to_cleared, is_cleared
  in_transition_from_game_to_cleared = False
  is_cleared = True
  set_overlay(Cleared(score))

def show_ranking_after_clear():
  global ranking_data, is_cleared
  is_cleared = False
  if os.path.exists(RANKING_FILENAME):
    ranking_data = pd.read_csv(RANKING_FILENAME).sort_values(by='score', ascending=False)
  else:
    ranking_data = None
  set_overlay(RankingAfterClear())

def input_name_for_ranking():
  set_overlay(InputName())

def show_start_menu():
  global is_gameover, in_start_menu
  is_gameover = False
  in_start_menu = True
  set_overlay(StartMenu())
  init_objects()

# --------------------------------------------------------------------------
# OpenGL functions
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

def gl_prepare_for_2D():
  glPushMatrix()
  glLoadIdentity()
  glMatrixMode(GL_PROJECTION)
  glPushMatrix()
  glLoadIdentity()
  glOrtho(-win.width//2, win.width//2, -win.height//2, win.height//2, -1, 1);
  glDisable(GL_LIGHTING)
  glDisable(GL_DEPTH_TEST)

def gl_prepare_for_3D():
  glEnable(GL_DEPTH_TEST)
  glEnable(GL_LIGHTING)
  glPopMatrix()
  glMatrixMode(GL_MODELVIEW)
  glPopMatrix()

def gl_clear_color_setting():
  global score
  if in_transition_from_game_to_cleared:
    factor = min(LIGHT_FLASH_MAX_ENERGY, score)/LIGHT_FLASH_MAX_ENERGY
    light_enhancement = factor*np.exp(-(t_transition/CLEAR_DELAY)*5)
    glClearColor(light_enhancement, light_enhancement, light_enhancement, light_enhancement)
  else:
    glClearColor(0, 0, 0, 0)

def gl_set_viewpoint():
  if in_transition_from_start_menu_to_game:
    x = (VIEW_START_X-VIEW_GAME_X)*np.exp(-0.05*np.abs(VIEW_GAME_X-VIEW_START_X)*t_transition/START_GAME_DELAY)+VIEW_GAME_X
    y = (VIEW_START_Y-VIEW_GAME_Y)*np.exp(-0.05*np.abs(VIEW_GAME_Y-VIEW_START_Y)*t_transition/START_GAME_DELAY)+VIEW_GAME_Y
    z = (VIEW_START_Z-VIEW_GAME_Z)*np.exp(-0.05*np.abs(VIEW_GAME_Z-VIEW_START_Z)*t_transition/START_GAME_DELAY)+VIEW_GAME_Z
    gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, -1, 0)
  elif in_transition_from_game_to_cleared:
    if t_transition < 0.5*CLEAR_DELAY:
      gluLookAt(VIEW_GAME_X, VIEW_GAME_Y, VIEW_GAME_Z, 0.0, 0.0, 0.0, 0.0, -1, 0)
    else:
      t = t_transition-0.5*CLEAR_DELAY
      x = (VIEW_GAME_X-VIEW_START_X)*np.exp(-0.1*np.abs(VIEW_START_X-VIEW_GAME_X)*t/CLEAR_DELAY)+VIEW_START_X
      y = (VIEW_GAME_Y-VIEW_START_Y)*np.exp(-0.1*np.abs(VIEW_START_Y-VIEW_GAME_Y)*t/CLEAR_DELAY)+VIEW_START_Y
      z = (VIEW_GAME_Z-VIEW_START_Z)*np.exp(-0.1*np.abs(VIEW_START_Z-VIEW_GAME_Z)*t/CLEAR_DELAY)+VIEW_START_Z
      gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, -1, 0)
  elif in_game or is_gameover:
    gluLookAt(VIEW_GAME_X, VIEW_GAME_Y, VIEW_GAME_Z, 0.0, 0.0, 0.0, 0.0, -1, 0)
  else:
    gluLookAt(VIEW_START_X, VIEW_START_Y, VIEW_START_Z, 0.0, 0.0, 0.0, 0.0, -1, 0)

# --------------------------------------------------------------------------
# Create window
# --------------------------------------------------------------------------

win = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption='HHG Game')

@win.event
def on_draw():
  gl_clear_color_setting()
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()
  gl_set_viewpoint()
  glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 0.0])
  mesh.draw()
  light_cone.draw()
  electron_ionized.draw()
  electron_localized.draw()
  nuclear.draw()
  gl_prepare_for_2D()
  if overlay:
    overlay.draw()
  if in_game:
    electric_field.draw()
  gl_prepare_for_3D()

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

in_start_menu = True
in_game = False
in_tutorial = False
in_transition_from_start_menu_to_game = False
is_ionized = False
is_gameover = False
in_transition_from_game_to_cleared = False
is_cleared = False

t_transition = 0
score = 0

# --------------------------------------------------------------------------
# Game update
# --------------------------------------------------------------------------

def update(dt):
  global t_transition, electric_field, mesh
  if in_game:
    check_gameover()
    check_collision()
  dt_scaled = dt*TIME_SCALE_FACTOR
  t_transition += dt
  electron_ionized.update(dt_scaled, [electric_field.ex, electric_field.ey])
  electron_localized.update(dt_scaled)
  nuclear.update()

pyglet.clock.schedule_interval(update, 1/60.)

# --------------------------------------------------------------------------
# Start game
# --------------------------------------------------------------------------

overlay = StartMenu()
set_overlay(overlay)
init_objects()
init_gl()
pyglet.app.run()
