# effects/glasses.py
import os
from pathlib import Path
import cv2
import numpy as np
from .effect_base import EffectBase

class GlassesEffect(EffectBase):
    """
    Procedural sunglasses overlay (no external image files).
    Draws dark lenses + colored frames inside the detected face box.
    Optionally uses a provided FaceDetector for fallback detection if faces list is empty.
    """
    name = "glasses"

    def __init__(self, detector=None, spacing: float = 1.00):
        """
        detector: optional FaceDetector instance (from core.face_detector). If provided and
                  faces==[], we will try detector.detect(gray) as a fallback.
        spacing : horizontal spacing multiplier between lenses (0.6 .. 1.6).
        """
        self.detector = detector
        self.spacing = float(np.clip(spacing, 0.6, 1.6))

    # ------------ internal helpers ------------
    @staticmethod
    def _alpha_paste(dst, src, mask, x, y):
        """Alpha blend src (BGR) onto dst (BGR) at (x, y) using mask in [0..1]."""
        h, w = src.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
        if x0 >= x1 or y0 >= y1:
            return
        roi_dst = dst[y0:y1, x0:x1].astype(np.float32)
        roi_src = src[(y0 - y):(y1 - y), (x0 - x):(x1 - x)].astype(np.float32)
        a = mask[(y0 - y):(y1 - y), (x0 - x):(x1 - x)][..., None].astype(np.float32)
        blended = roi_src * a + roi_dst * (1 - a)
        dst[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    def _draw_glasses_in_facebox(self, out_bgr, original_bgr, face_rect):
        """Draws the sunglasses procedurally inside the given face rectangle."""
        x, y, fw, fh = face_rect
        H, W = out_bgr.shape[:2]

        # Dim the frame to make the face area pop
        out_bgr[:] = (out_bgr * 0.6).astype(np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + fw), min(H, y + fh)
        out_bgr[y0:y1, x0:x1] = original_bgr[y0:y1, x0:x1]

        # Overlay canvas limited to the face box
        overlay_h, overlay_w = fh, fw
        overlay = np.zeros((overlay_h, overlay_w, 3), np.uint8)
        alpha   = np.zeros((overlay_h, overlay_w), np.float32)

        # --- Geometry (adjustable spacing) ---
        # Lens centers (relative to face box)
        eye_y  = int(0.40 * fh)
        eye_rx = int(0.16 * fw)
        eye_ry = int(0.14 * fh)

        # Base positions (before spacing)
        base_left_cx  = int(0.27 * fw)
        base_right_cx = int(0.73 * fw)

        # Apply spacing around face center
        cx = fw * 0.5
        left_cx  = int(cx - (cx - base_left_cx)  * self.spacing)
        right_cx = int(cx + (base_right_cx - cx) * self.spacing)

        # Style
        color_lens       = (40, 40, 40)     # dark gray
        color_frame      = (0, 200, 255)    # cyan/orange-ish frame
        frame_thickness  = 2
        bridge_thickness = 3
        alpha_value      = 0.85

        # Lenses (filled)
        cv2.ellipse(overlay, (left_cx,  eye_y), (eye_rx, eye_ry), 0, 0, 360, color_lens, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (right_cx, eye_y), (eye_rx, eye_ry), 0, 0, 360, color_lens, -1, cv2.LINE_AA)

        # Frames (strokes)
        cv2.ellipse(overlay, (left_cx,  eye_y), (eye_rx, eye_ry), 0, 0, 360, color_frame, frame_thickness, cv2.LINE_AA)
        cv2.ellipse(overlay, (right_cx, eye_y), (eye_rx, eye_ry), 0, 0, 360, color_frame, frame_thickness, cv2.LINE_AA)

        # Bridge
        cv2.line(overlay,
                 (left_cx + eye_rx - 5,  eye_y),
                 (right_cx - eye_rx + 5, eye_y),
                 color_frame, bridge_thickness, cv2.LINE_AA)

        # Alpha where we drew
        alpha[overlay.sum(axis=2) > 0] = alpha_value

        # Paste into output
        self._alpha_paste(out_bgr, overlay, alpha, x, y)

        # HUD label
        cv2.putText(out_bgr, "Freestyle: Sunglasses (no PNG)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------ public API ------------
    def nudge_spacing(self, delta: float):
        self.spacing = float(np.clip(self.spacing + delta, 0.6, 1.6))

    def apply(self, frame_bgr, frame_gray, faces):
        if not self.enabled:
            return frame_bgr

        out = frame_bgr.copy()
        H, W = out.shape[:2]

        # Use faces from upstream; if none, fallback to detector if provided
        faces_in = list(faces) if faces is not None and len(faces) else []
        if not faces_in and self.detector is not None:
            faces_in = list(self.detector.detect(frame_gray))

        if not faces_in:
            # Graceful fallback: light vignette so the frame isn't empty
            yy, xx = np.mgrid[0:H, 0:W]
            cx, cy = W * 0.5, H * 0.5
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            vign = (1 - (r / r.max())**2 * 0.45).clip(0.7, 1.0).astype(np.float32)
            return (out.astype(np.float32) * vign[..., None]).astype(np.uint8)

        # Use the largest face
        x, y, fw, fh = max(faces_in, key=lambda b: b[2] * b[3])
        self._draw_glasses_in_facebox(out, frame_bgr, (x, y, fw, fh))
        return out
