# effects/motion_interact.py
import cv2
import numpy as np
from .effect_base import EffectBase

def _map(val, a0, a1, b0, b1):
    t = 0.0 if a1 == a0 else (val - a0) / (a1 - a0)
    t = max(0.0, min(1.0, t))
    return b0 + (b1 - b0) * t

class MotionInteraction(EffectBase):
    """
    Motion-based hand tracker using background subtraction + contour analysis.
    - Tracks the largest moving blob (your hand) and estimates finger count via convexity defects.
    - Exposes normalized hand position (tx, ty) and finger_count.
    - Also draws a simple 'hat' above the detected face that follows the hand (demo interaction).
    """
    name = "interaction"

    def __init__(self, roi="right", min_area=2500):
        """
        roi: 'full' | 'right' | 'left'  -> region of interest for hand
        min_area: minimum contour area to accept as a hand
        """
        self.bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.roi_mode = roi
        self.min_area = int(min_area)

        # outputs
        self.cx = None
        self.cy = None
        self.fingers = 0
        self.box = None  # (x,y,w,h) of detected hand region (in full-frame coords)

    # ---------------- low-level tracking ----------------
    def _roi_rect(self, W, H):
        if self.roi_mode == "right":
            return (W*2//3, 0, W//3, H)
        if self.roi_mode == "left":
            return (0, 0, W//3, H)
        return (0, 0, W, H)

    def _segment_moving(self, frame_bgr, rect):
        x,y,w,h = rect
        roi = frame_bgr[y:y+h, x:x+w]
        fg = self.bg.apply(roi)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        return mask, x, y

    def _largest_contour(self, mask, offset_x, offset_y):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return None
        x,y,w,h = cv2.boundingRect(c)
        # global coords
        self.box = (x+offset_x, y+offset_y, w, h)
        M = cv2.moments(c)
        if M["m00"] > 1e-3:
            self.cx = int(M["m10"]/M["m00"]) + offset_x
            self.cy = int(M["m01"]/M["m00"]) + offset_y
        else:
            self.cx = self.cy = None
        return c

    def _estimate_fingers(self, contour):
        """Very simple convexity-defects-based finger counter."""
        self.fingers = 0
        if contour is None or len(contour) < 5:
            return 0
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        cnt = 0
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i, 0]
            start = contour[s][0]; end = contour[e][0]; far = contour[f][0]
            # basic geometry thresholds
            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(far - end)
            if b*c == 0:
                continue
            # angle at the defect point
            cos_angle = (b*b + c*c - a*a) / (2*b*c + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            # count as a finger gap if angle is sharp and depth is decent
            if angle < 90.0 and depth > 1000:  # depth is in fixed-point (8.24)
                cnt += 1
        # rough mapping: gaps ~ fingers-1
        self.fingers = int(np.clip(cnt + 1, 0, 5))
        return self.fingers

    # ---------------- interaction render ----------------
    def _draw_hat_over_face(self, frame_bgr, face_rect, tx):
        """
        Draw a simple triangle 'hat' above the face.
        Position shifts horizontally with tx in [0..1].
        """
        if face_rect is None:
            return
        x, y, w, h = face_rect
        # target position: above forehead, horizontally shifted by tx
        base_cx = x + w//2
        shift = int(_map(tx if tx is not None else 0.5, 0, 1, -w*0.3, w*0.3))
        cx = base_cx + shift
        top = (cx, max(0, y - int(0.45*h)))
        left = (cx - int(0.28*w), y - int(0.15*h))
        right = (cx + int(0.28*w), y - int(0.15*h))
        pts = np.array([top, left, right], dtype=np.int32)
        overlay = frame_bgr.copy()
        cv2.fillConvexPoly(overlay, pts, (50, 200, 50))
        cv2.polylines(overlay, [pts], True, (255,255,255), 2, cv2.LINE_AA)
        # alpha blend
        alpha = 0.6
        frame_bgr[:] = (alpha*overlay + (1-alpha)*frame_bgr).astype(np.uint8)

    # ---------------- public API ----------------
    def get_controls(self, width, height):
        """Return (tx, ty, fingers) in [0..1]×[0..1] and 0..5. None if not tracked."""
        if self.cx is None or self.cy is None:
            return None, None, 0
        tx = _map(self.cx, 0, max(1, width-1), 0.0, 1.0)
        ty = _map(self.cy, 0, max(1, height-1), 0.0, 1.0)
        return tx, ty, self.fingers

    def apply(self, frame_bgr, frame_gray, faces):
        if not self.enabled:
            return frame_bgr

        H, W = frame_bgr.shape[:2]
        rx, ry, rw, rh = self._roi_rect(W, H)
        mask, ox, oy = self._segment_moving(frame_bgr, (rx, ry, rw, rh))
        cnt = self._largest_contour(mask, ox, oy)
        self._estimate_fingers(cnt)

        # Visualize ROI and detection
        cv2.rectangle(frame_bgr, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
        if self.box:
            x,y,w,h = self.box
            cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (0,180,255), 2)
        if self.cx is not None and self.cy is not None:
            cv2.circle(frame_bgr, (self.cx, self.cy), 6, (0,0,255), -1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"fingers:{self.fingers}", (rx+8, ry+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        # Demo interaction: draw a hat over the largest face that follows hand x
            # Demo interaction: draw a hat over the largest face that follows hand x
        faces_list = []
        if faces is not None:
            try:
                # works for np.ndarray or list
                if len(faces) > 0:
                    faces_list = list(faces)
            except TypeError:
                faces_list = list(faces)

        if len(faces_list) > 0:
            face = max(faces_list, key=lambda r: r[2] * r[3])
            tx, ty, _ = self.get_controls(W, H)
            self._draw_hat_over_face(frame_bgr, face, tx)

        return frame_bgr

