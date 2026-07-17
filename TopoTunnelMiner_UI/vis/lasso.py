"""
Lasso (freehand polygon) selection overlay and geometry utilities.
Provides a transparent paint overlay for drawing the lasso shape,
plus vectorized projection and point-in-polygon routines for fast selection.
"""
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF


class LassoOverlay(QWidget):
    """Transparent paint-only overlay for lasso visualization.

    Must be created as a child of the VTK render widget so that it
    covers the viewport exactly.  Installs an event filter on its
    parent to stay in sync on resize.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self._points: list[tuple[float, float]] = []
        self._closed = False
        if parent is not None:
            parent.installEventFilter(self)

    # ── geometry sync ───────────────────────────────────
    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() == QEvent.Type.Resize:
            self.resize(obj.size())
        return False

    # ── public API ──────────────────────────────────────
    def set_points(self, points: list[tuple[float, float]], closed: bool = False):
        self._points = points
        self._closed = closed
        self.update()

    def clear(self):
        self._points = []
        self._closed = False
        self.update()

    # ── painting ────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Always clear previous frame first. On translucent overlay widgets,
        # returning early without clearing leaves stale pixels on screen.
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        if len(self._points) < 2:
            painter.end()
            return

        pen = QPen(QColor(65, 105, 225, 200), 2, Qt.SolidLine)
        painter.setPen(pen)

        if self._closed and len(self._points) >= 3:
            painter.setBrush(QColor(65, 105, 225, 30))
            poly = QPolygonF([QPointF(x, y) for x, y in self._points])
            painter.drawPolygon(poly)
        else:
            for i in range(len(self._points) - 1):
                x0, y0 = self._points[i]
                x1, y1 = self._points[i + 1]
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        painter.end()


# ── vectorized projection ──────────────────────────────

def project_points_to_screen(
    points_3d: np.ndarray,
    renderer,
    widget_w: int,
    widget_h: int,
) -> np.ndarray:
    """Project Nx3 world points → Nx2 widget-space screen coords.

    Uses the VTK camera composite projection matrix for accuracy,
    then maps NDC [-1,1] → widget logical coordinates so the result
    lives in the same space as Qt mouse positions.
    """
    N = len(points_3d)
    if N == 0:
        return np.empty((0, 2), dtype=np.float64)

    camera = renderer.GetActiveCamera()
    aspect = widget_w / max(widget_h, 1)
    near, far = camera.GetClippingRange()

    vtk_mat = camera.GetCompositeProjectionTransformMatrix(aspect, near, far)
    mat = np.array(
        [[vtk_mat.GetElement(i, j) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )

    # Homogeneous world coords → clip coords
    pts_h = np.ones((N, 4), dtype=np.float64)
    pts_h[:, :3] = points_3d
    clip = (mat @ pts_h.T).T  # Nx4

    w = clip[:, 3]
    valid = np.abs(w) > 1e-10

    ndc = np.zeros((N, 2), dtype=np.float64)
    ndc[valid, 0] = clip[valid, 0] / w[valid]
    ndc[valid, 1] = clip[valid, 1] / w[valid]

    # NDC → widget coords (Qt origin = top-left)
    screen = np.full((N, 2), 1e30, dtype=np.float64)
    screen[valid, 0] = (ndc[valid, 0] + 1.0) * 0.5 * widget_w
    screen[valid, 1] = widget_h - (ndc[valid, 1] + 1.0) * 0.5 * widget_h

    # Points behind the camera get sentinel value → never inside any polygon
    screen[w <= 0] = 1e30

    return screen


# ── vectorized ray-casting PIP ─────────────────────────

def points_in_polygon(
    points: np.ndarray,   # Nx2 screen coords
    polygon: np.ndarray,  # Mx2 polygon vertices
) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test.

    Casts a horizontal ray to the right from each point and counts
    edge crossings.  Fully vectorized over *points* (loop only over
    polygon edges, which are typically 20-100).

    Returns bool array of length N.
    """
    N = len(points)
    M = len(polygon)
    if N == 0 or M < 3:
        return np.zeros(N, dtype=bool)

    px = points[:, 0].copy()
    py = points[:, 1].copy()
    inside = np.zeros(N, dtype=bool)

    for i in range(M):
        j = (i - 1) % M
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])

        # Does horizontal ray from (px, py) → +x cross this edge?
        cond = (yi > py) != (yj > py)
        if not np.any(cond):
            continue

        dy = yj - yi
        if abs(dy) < 1e-30:
            continue

        t = (py - yi) / dy
        x_cross = xi + t * (xj - xi)
        cross = cond & (px < x_cross)
        inside ^= cross

    return inside
