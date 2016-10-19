import pyglet
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

window = pyglet.window.Window()
vertices = [0, 0, window.width, 0, window.width, window.height]
vertices_gl = (GLfloat*len(vertices))(*vertices)
a = 0

def init():
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
  glEnable(GL_CULL_FACE)
  glCullFace(GL_BACK)
  glEnable(GL_DEPTH_TEST)

def update(dt):
  global a
  a += 0.1

def draw_sphere(position, color, size):
  glTranslated(position[0], position[1], position[2])
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1, 1, 1, 1])
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 40)
  glutSolidSphere(size, 15, 10)

@window.event
def on_draw():
  global a
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()
  gluLookAt(0, 0, 30, 0.0, 0.0, 0.0, 0.0, 1, 0)
  glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 0.0])
  glPushMatrix()
  draw_sphere([0, 0, 0], [1, 0, 0, 0], 3)
  glPopMatrix()
  draw_sphere([10*np.sin(a), 10*np.cos(a), 0], [0, 1, 0, 0], 1)

@window.event
def on_resize(width, height):
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
  glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glViewport(0, 0, width, height)
  gluPerspective(45, window.width//window.height, 0.1, 1000)
  glMatrixMode(GL_MODELVIEW)
  return pyglet.event.EVENT_HANDLED

@window.event
def on_key_press(symbol, modifiers):
  if symbol == pyglet.window.key.A:
    global a
    a += 1

init()
pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()
