import pandas as pd

from typing import List

from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from numpy import ndarray
from scipy.interpolate import BSpline
import scipy.interpolate as interpolate

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

colors = {
    "grid": [153/256, 153/256, 153/256],
    "axis": [32/256, 34/256, 33/256],
    "line": [252/256, 33/256, 33/256],
    "point": [91/256, 176/256, 122/256],
}


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.x}, {self.y}"


class Spline:

    def __init__(self, min_pts=2):
        self.container = []
        self.min_pts = min_pts
        self.pts_size = 10.
        self.position = np.array([0., 0., 0.])
        self.scale = 1.

    def add_point(self, x, y):
        self.container.append(Point(x, y))
        self.container = sorted(self.container, key=lambda p: p.x)

    def draw(self):
        if len(self.container) > 2:
            self._line()
        self._pts()

    def scaling(self, arg):
        self.scale = arg

        for i, _ in enumerate(self.container):
            self.container[i].x *= self.scale
            self.container[i].y *= self.scale

    def _line(self):
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(*self.position)
        glLineWidth(2.2)
        glColor3f(*colors["line"])

        points_x_a = np.array([point.x for point in self.container])
        points_y_a = np.array([point.y for point in self.container])

        # append the starting x,y coordinates
        points_x_a = np.r_[points_x_a, points_x_a[0]]
        points_y_a = np.r_[points_y_a, points_y_a[0]]

        tck_u = interpolate.splprep(
            [points_x_a, points_y_a],
            k=3,
            s=0,
            per=True,
        )
        tck, u = tck_u[0], tck_u[1]

        n = 1000
        xx = np.linspace(-1, 1, n)
        xi, yi = interpolate.splev(xx, tck, ext=3)
        glBegin(GL_LINE_STRIP)

        for x, y in zip(xi, yi):
            glVertex2f(x, y)

        glEnd()
        glPopMatrix()

    def _pts(self):
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(*self.position)
        glPointSize(self.pts_size)
        glColor3f(*colors["point"])
        glBegin(GL_POINTS)

        for point in self.container:
            glVertex2f(point.x, point.y)

        glEnd()
        glPopMatrix()


class Window:
    def __init__(self, width, height, caption):
        self.width = width
        self.height = height
        self.caption = caption
        self.spline = Spline()
        self.center = np.array([0., 0.])
        self.coords = np.array([-1., -1.])
        self.labels_x = np.arange(-1.0, 1.1, 0.1)
        self.labels_y = np.arange(-1.0, 1.1, 0.1)
        self.points_read = False

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(self.caption)
        glClearColor(1, 1, 1, 0.)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

        glutReshapeFunc(self._reshape)
        glutDisplayFunc(self._render_display)
        glutMouseFunc(self._on_click)
        glutKeyboardFunc(self._on_press)
        glutSpecialFunc(self._on_press)

        glutMainLoop()

    def _render_display(self):
        glClear(GL_COLOR_BUFFER_BIT)

        if len(self.spline.container) < 3:
            self._render_string(-0.2, -0.5, "Not enough points", colors["line"], 2)
        self._draw_grid()
        self._draw_axis()
        self.spline.draw()
        glutSwapBuffers()

    def _draw_axis(self):
        glPushMatrix()
        o = self.center
        glColor3f(*colors["axis"])
        glBegin(GL_LINES)
        glVertex2f(o[0], -1)
        glVertex2f(o[0], 1)
        glVertex2f(-1, o[1])
        glVertex2f(1, o[1])
        glEnd()
        self._render_string(o[0] + 0.02, .96, "y", colors["axis"], 1)
        self._render_string(.96, o[1] + 0.02, "x", colors["axis"], 1)
        self._draw_axis_labels()
        glPopMatrix()

    def _draw_axis_labels(self):
        x_coord = np.arange(-1.0, 1.1, 0.1)
        for x_, x in zip(self.labels_x, x_coord):
            self._render_string(x - 0.03, -0.96, str(np.round(x_, 2)),
                                [0., 0., 0.], 1)
        y_coord = np.arange(-1.0, 1.1, 0.1)
        for y_, y in zip(self.labels_y, y_coord):
            self._render_string(-1., y - 0.01, str(np.round(y_, 2)),
                                [0., 0., 0.], 1)

    def _reshape(self, width, height):
        margin, tick_size = 10, 20
        glViewport(
            margin + tick_size,
            margin + tick_size,
            self.width - margin * 2 - tick_size,
            self.height - margin * 2 - tick_size
        )

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _on_click(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            x = (2.0 * x / self.width - 1.0)
            y = (1.0 - 2.0 * y / self.height)
            self.spline.add_point(x, y)

        glutPostRedisplay()

    def _on_press(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.spline.position += np.array([-.1, 0, 0])
            # self.center += np.array([-.1, 0])
            self.coords += np.array([-.1, 0])
            self.labels_x += -1
        elif key == GLUT_KEY_RIGHT:
            self.spline.position += np.array([.1, 0, 0])
            # self.center += np.array([.1, 0])
            self.coords += np.array([.1, 0])
            self.labels_x += 1
        elif key == GLUT_KEY_UP:
            self.spline.position += np.array([0, .1, 0])
            # self.center += np.array([0, .1])
            self.coords += np.array([0, .1])
            self.labels_y += 1
        elif key == GLUT_KEY_DOWN:
            self.spline.position += np.array([0, -.1, 0])
            # self.center += np.array([0, -.1])
            self.coords += np.array([0, -.1])
            self.labels_y += -1
        elif key == b'=':
            self.spline.scaling(2)
            self.coords *= 0.5
            self.labels_x *= 0.5
            self.labels_y *= 0.5
        elif key == b'-':
            self.spline.scaling(0.5)
            self.coords *= 2
            self.labels_x *= 2
            self.labels_y *= 2

        elif key == b'r':
            self.spline.container = []

            self.points_read = False

        elif key == b"f":
            if not self.points_read:
                df: pd.DataFrame = pd.read_csv('data/data.csv')
                df.apply(lambda row: self.spline.add_point(row.x, row.y), axis=1)

                self.points_read = True

        glutPostRedisplay()

    @staticmethod
    def _render_string(x, y, string: str, color, mode):
        glColor3f(*color)
        # glRasterPos2f(-.99, -0.96)
        glRasterPos2f(x, y)
        for c in string:
            if mode == 1:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
            else:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(c))

    @staticmethod
    def _draw_grid():
        glPushMatrix()
        glLoadIdentity()
        glLineWidth(1.)
        glColor3f(*colors["grid"])
        glBegin(GL_LINES)
        for i in np.arange(-1.0, 1.1, 0.1):
            glVertex2f(-1.0, i)
            glVertex2f(1.0, i)
        for i in np.arange(-1.0, 1.1, 0.1):
            glVertex2f(i, -1.0)
            glVertex2f(i, 1.0)
        glEnd()
        glPopMatrix()



if __name__ == '__main__':
    Window(800, 600, "RR").run()