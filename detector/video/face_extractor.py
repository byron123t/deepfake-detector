"""
Face detection and cropping.

Uses MediaPipe BlazeFace (Short-Range) — a sub-millisecond face detector
designed for mobile/real-time use (Bazarevsky et al., 2019).  Falls back
to OpenCV's DNN-based face detector if MediaPipe is unavailable.

Why not MTCNN or RetinaFace?
  MTCNN is slower (~80 ms/frame) and RetinaFace heavier still.
  BlazeFace runs in <1 ms on CPU for typical laptop resolutions, which is
  critical since the downstream deepfake classifier is already the bottleneck.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("mediapipe not installed. Run: pip install mediapipe")


class FaceExtractor:
    """
    Detects faces in a frame and returns cropped face patches.

    Parameters
    ----------
    min_confidence : float
        Minimum detection score to accept a bounding box.
    padding : float
        Fractional padding to add around the tight bounding box.
    output_size : int
        Both spatial dimensions of the returned square crop.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        padding: float = 0.2,
        output_size: int = 224,
    ) -> None:
        self.min_confidence = min_confidence
        self.padding = padding
        self.output_size = output_size
        self._detector = self._build_detector()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_detector(self):
        if _MEDIAPIPE_AVAILABLE:
            return mp.solutions.face_detection.FaceDetection(
                model_selection=0,           # short-range model (< 2 m from camera)
                min_detection_confidence=self.min_confidence,
            )
        logger.warning(
            "Falling back to OpenCV Haar cascade face detector (less accurate)."
        )
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _boxes_mediapipe(
        self, frame_rgb: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        results = self._detector.process(frame_rgb)
        if not results.detections:
            return []

        h, w = frame_rgb.shape[:2]
        boxes = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            boxes.append((x1, y1, bw, bh))
        return boxes

    def _boxes_opencv(
        self, frame_gray: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        detections = self._detector.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(detections) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    def _pad_and_crop(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        img_h, img_w = frame.shape[:2]
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (self.output_size, self.output_size))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_faces(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Return a list of face crops (BGR, uint8, output_size×output_size).
        Returns an empty list if no face is found.
        """
        if _MEDIAPIPE_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes = self._boxes_mediapipe(frame_rgb)
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self._boxes_opencv(gray)

        crops = []
        for x, y, w, h in boxes:
            crop = self._pad_and_crop(frame_bgr, x, y, w, h)
            crops.append(crop)

        return crops

    def extract_largest_face(
        self, frame_bgr: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return only the largest (by area) detected face, or None."""
        if _MEDIAPIPE_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes = self._boxes_mediapipe(frame_rgb)
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self._boxes_opencv(gray)

        if not boxes:
            return None

        # Pick largest by w*h
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = boxes[0]
        return self._pad_and_crop(frame_bgr, x, y, w, h)

    def close(self) -> None:
        if _MEDIAPIPE_AVAILABLE and hasattr(self._detector, "close"):
            self._detector.close()
