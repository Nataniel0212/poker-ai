"""
Screen capture module — two modes:
  A) Direct screen capture via mss (for development/testing)
  B) Camera-based capture via OpenCV (for undetectable live play)
"""

import numpy as np
import cv2
import time
from abc import ABC, abstractmethod


class BaseCapture(ABC):
    """Base class for all capture methods."""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture a frame and return as BGR numpy array."""
        pass

    @abstractmethod
    def release(self):
        """Release resources."""
        pass


class ScreenCapture(BaseCapture):
    """Direct screen capture using mss — fast, for development.

    Uses thread-local mss instances since Windows GDI device contexts
    cannot be shared across threads.
    """

    def __init__(self, monitor_region: dict = None):
        """
        Args:
            monitor_region: dict with keys 'top', 'left', 'width', 'height'
                            If None, captures primary monitor.
        """
        import threading
        self._monitor_region = monitor_region
        self._local = threading.local()

    def _get_sct(self):
        """Get or create a thread-local mss instance."""
        if not hasattr(self._local, 'sct'):
            import mss
            self._local.sct = mss.mss()
        return self._local.sct

    @property
    def region(self):
        if self._monitor_region:
            return self._monitor_region
        sct = self._get_sct()
        return sct.monitors[1]

    def set_region(self, top: int, left: int, width: int, height: int):
        """Update the capture region (e.g. to match poker client window)."""
        self._monitor_region = {
            "top": top,
            "left": left,
            "width": width,
            "height": height,
        }

    def capture(self) -> np.ndarray:
        """Capture screen region and return as BGR numpy array."""
        sct = self._get_sct()
        screenshot = sct.grab(self.region)
        frame = np.array(screenshot)
        # mss returns BGRA, convert to BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def release(self):
        if hasattr(self._local, 'sct'):
            self._local.sct.close()


class CameraCapture(BaseCapture):
    """Camera-based capture — reads from webcam pointed at another screen.
    Undetectable by poker clients since it's a completely separate device."""

    def __init__(self, camera_index: int = 0, calibration_file: str = None):
        """
        Args:
            camera_index: Which camera to use (0 = default)
            calibration_file: Path to saved perspective transform matrix
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        # Set high resolution if available
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.transform_matrix = None
        self.output_size = (1920, 1080)

        if calibration_file:
            self.load_calibration(calibration_file)

    def calibrate(self, frame: np.ndarray = None):
        """Interactive calibration — user clicks 4 corners of the poker client
        on the monitored screen. Returns perspective transform matrix."""
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera")

        print("=== CAMERA CALIBRATION ===")
        print("Click the 4 corners of the poker table area on the camera feed.")
        print("Order: top-left, top-right, bottom-right, bottom-left")

        points = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                print(f"  Point {len(points)}: ({x}, {y})")
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Calibration", frame)

        cv2.imshow("Calibration", frame)
        cv2.setMouseCallback("Calibration", click_handler)

        while len(points) < 4:
            cv2.waitKey(100)

        cv2.destroyWindow("Calibration")

        src_pts = np.float32(points)
        dst_pts = np.float32([
            [0, 0],
            [self.output_size[0], 0],
            [self.output_size[0], self.output_size[1]],
            [0, self.output_size[1]],
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return self.transform_matrix

    def save_calibration(self, filepath: str):
        """Save calibration matrix to file."""
        if self.transform_matrix is not None:
            np.save(filepath, self.transform_matrix)

    def load_calibration(self, filepath: str):
        """Load calibration matrix from file."""
        self.transform_matrix = np.load(filepath)

    def capture(self) -> np.ndarray:
        """Capture frame from camera, apply perspective correction if calibrated."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read from camera")

        if self.transform_matrix is not None:
            frame = cv2.warpPerspective(
                frame, self.transform_matrix, self.output_size
            )

        return frame

    def release(self):
        self.cap.release()


class CaptureLoop:
    """Continuously captures frames at a target rate and provides the latest frame."""

    def __init__(self, capturer: BaseCapture, target_fps: float = 2.0):
        """
        Args:
            capturer: A ScreenCapture or CameraCapture instance
            target_fps: How many frames per second to capture
        """
        self.capturer = capturer
        self.interval = 1.0 / target_fps
        self.latest_frame = None
        self.last_capture_time = 0

    def update(self) -> np.ndarray:
        """Capture a new frame if enough time has passed. Returns latest frame."""
        now = time.time()
        if now - self.last_capture_time >= self.interval:
            self.latest_frame = self.capturer.capture()
            self.last_capture_time = now
        return self.latest_frame

    def get_frame(self) -> np.ndarray:
        """Get the latest captured frame without capturing a new one."""
        return self.latest_frame
