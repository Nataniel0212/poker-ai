"""Skarmfangst och regionval.

mss ar inte tradsakert — dess GDI-kontext maste skapas i den trad som anvander
den. Det var en av kraascherna i det gamla projektet, sa har halls instansen
trad-lokal fran borjan.
"""

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

_local = threading.local()


def _grabber():
    """En mss-instans per trad."""
    if getattr(_local, "sct", None) is None:
        import mss
        _local.sct = mss.mss()
    return _local.sct


def screen_size() -> Tuple[int, int]:
    monitor = _grabber().monitors[0]
    return monitor["width"], monitor["height"]


def grab(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Fanga skarmen (eller en region) som BGR-bild."""
    sct = _grabber()
    if region is None:
        box = sct.monitors[0]
    else:
        x, y, w, h = region
        box = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
    raw = np.asarray(sct.grab(box))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def select_region(window_title: str = "Dra en ruta runt pokerbordet") -> Optional[Tuple[int, int, int, int]]:
    """Lat anvandaren dra ut en region pa en frusen skarmbild.

    Returnerar (x, y, w, h) i skarmkoordinater, eller None om det avbryts.
    """
    frame = grab()
    screen_h, screen_w = frame.shape[:2]

    # Krymp till nagot som sakert far plats pa skarmen
    max_w, max_h = int(screen_w * 0.85), int(screen_h * 0.85)
    scale = min(1.0, max_w / screen_w, max_h / screen_h)
    preview = cv2.resize(frame, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    state = {"start": None, "end": None, "done": False, "dragging": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"] = (x, y)
            state["dragging"] = False
            state["done"] = True

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)

    help_text = "Dra en ruta. Enter = klar, Esc = avbryt, r = gor om"
    while True:
        canvas = preview.copy()
        cv2.putText(canvas, help_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3)
        cv2.putText(canvas, help_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        if state["start"] and state["end"]:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 220, 0), 2)
        cv2.imshow(window_title, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:                      # Esc
            cv2.destroyWindow(window_title)
            return None
        if key in (13, 10) or (state["done"] and key == 255 and False):
            break
        if key in (ord("r"), ord("R")):
            state.update(start=None, end=None, done=False, dragging=False)
        if state["done"] and state["start"] and state["end"]:
            # Vanta pa Enter sa anvandaren kan justera med r
            pass

    cv2.destroyWindow(window_title)

    if not (state["start"] and state["end"]):
        return None

    (x0, y0), (x1, y1) = state["start"], state["end"]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    inv = 1.0 / scale if scale else 1.0
    region = (int(x0 * inv), int(y0 * inv),
              int((x1 - x0) * inv), int((y1 - y0) * inv))
    if region[2] < 40 or region[3] < 40:
        return None
    return region
