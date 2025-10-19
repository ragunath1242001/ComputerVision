# core/face_detector.py
import cv2
from pathlib import Path

# ---- Paths to Haar cascades (relative to project root) ----
ROOT = Path(__file__).resolve().parents[1]          # .../<project root>
HAAR_DIR = ROOT / "assets" / "haar"
FACE_XML = HAAR_DIR / "haarcascade_frontalface_default.xml"
EYE_XML  = HAAR_DIR / "haarcascade_eye.xml"

# ---- Load cascades exactly like your snippet, but from assets/haar ----
_face_cascade = cv2.CascadeClassifier(str(FACE_XML))
assert not _face_cascade.empty(), f"Failed to load face cascade: {FACE_XML}"

_eye_cascade = cv2.CascadeClassifier(str(EYE_XML))
assert not _eye_cascade.empty(), f"Failed to load eye cascade:  {EYE_XML}"


class FaceDetector:
    """
    Simple Haar-based detector (face + eyes).
    Parameters mirror your snippet (scaleFactor=1.3, minNeighbors=5).
    """

    def __init__(self, min_face: int = 60, min_eye: int = 20):
        self.min_face = (int(min_face), int(min_face))
        self.min_eye  = (int(min_eye), int(min_eye))

    # ---- Face detection on full grayscale frame ----
    def detect(self, frame_gray):
        """
        Returns list of (x, y, w, h) face rectangles in image coordinates.
        """
        return _face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=self.min_face
        )

    # ---- Eye detection inside a face ROI (ROI coordinates) ----
    def detect_eyes(self, roi_gray):
        """
        Detect eyes inside a grayscale ROI.
        Returns list of (ex, ey, ew, eh) in ROI coordinates.
        """
        return _eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=self.min_eye
        )

    # ---- Eye detection but returned in GLOBAL coords for convenience ----
    def detect_eyes_in_face(self, frame_gray, face_rect):
        """
        Detect eyes inside a face rectangle and return boxes in GLOBAL image coords.
        face_rect: (x, y, w, h) in image coordinates.
        Returns list of (X, Y, W, H) in image coordinates.
        """
        x, y, w, h = face_rect
        roi_gray = frame_gray[y:y+h, x:x+w]
        eyes = self.detect_eyes(roi_gray)
        return [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]

