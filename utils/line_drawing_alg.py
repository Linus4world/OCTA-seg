"""Script demonstrating drawing of anti-aliased lines using Xiaolin Wu's line
algorithm

usage: python xiaolinwu.py [output-file]

"""
from __future__ import division
import numpy as np

from PIL import Image


def _fpart(x):
    return x - int(x)

def _rfpart(x):
    return 1 - _fpart(x)

def putpixel(img, xy, alpha=1):
    """Paints color over the background at the point xy in img.

    Use alpha for blending. alpha=1 means a completely opaque foreground.

    """
    x,y = xy
    # c = tuple(map(lambda bg, fg: int(round(alpha * fg + (1-alpha) * bg)),
    #               img.getpixel(xy), color))
    # img.putpixel(xy, c)
    x = min(x, img.shape[0]-1)
    y = min(y, img.shape[1]-1)
    img[x,y] = max(img[x,y],alpha)

def draw_line(img: np.ndarray, p1, p2):
    """Draws an anti-aliased line in img from p1 to p2 with the given color."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    if dx == 0 and dy==0:
        return
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px,py), (py,px))[steep]

    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2, p1, p2 = x2, x1, y2, y1, p2, p1

    grad = dy/dx
    intery = y1 + _rfpart(x1) * grad
    def draw_endpoint(pt):
        x, y = pt
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        putpixel(img, p(px, py), _rfpart(yend) * xgap)
        putpixel(img, p(px, py+1), _fpart(yend) * xgap)
        return px

    xstart = draw_endpoint(p(*p1)) + 1
    xend = draw_endpoint(p(*p2))

    for x in range(xstart, xend):
        y = int(intery)
        putpixel(img, p(x, y), _rfpart(intery))
        putpixel(img, p(x, y+1), _fpart(intery))
        intery += grad

