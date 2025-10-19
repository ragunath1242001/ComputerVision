# effects/face_warp.py
import cv2
import numpy as np
from .effect_base import EffectBase
from core.geometry import clamp_rect

class FaceWarp(EffectBase):
    """
    Strong, real-time face warp that follows the detected face.
    Combines bulge (alien), pinch (temples), twist (swirl), and vertical stretch.
    Tuned for dramatic results similar to fun-house filters.
    """
    name = "warp"

    def __init__(
        self,
        warp_strength: float = 0.75,   # overall intensity (0.2 .. 0.9)
        radius_scale: float = 0.65,    # radius as fraction of min(face_w, face_h)
        bulge: float = 1.20,           # + expands center, - pinches
        pinch: float = 0.80,           # extra pinch near edge of radius
        twist_radians: float = 1.20,   # total swirl at center (radians)
        smile_stretch: float = 0.90,   # vertical stretch near center
        feather: float = 0.20          # soft blend at face ROI edges (0.05..0.45)
    ):
        self.strength = float(warp_strength)
        self.radius_scale = float(radius_scale)
        self.bulge = float(bulge)
        self.pinch = float(pinch)
        self.twist = float(twist_radians)
        self.smile = float(smile_stretch)
        self.feather = float(np.clip(feather, 0.05, 0.45))

    # -------- core math --------
    def _maps_funhouse(self, h, w, cx, cy, R, strength):
        """
        Build remap matrices (float32) for combined bulge+pinch+twist+vertical stretch.
        Only pixels within radius R are warped; others map to identity.
        """
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx*dx + dy*dy)
        t = np.clip(r / (R + 1e-6), 0.0, 1.0)     # 0 at center, 1 at radius

        # Weights that fade to zero at radius
        w_center = (1.0 - t)**2               # center-focused
        w_edge   = (1.0 - t) * t              # mid/edge band

        # --- Bulge / Pinch (radial) ---
        # radial scale factor: >1 bulge at center, <1 pinch
        radial = 1.0 + strength * ( self.bulge * w_center - self.pinch * w_edge )
        radial = np.clip(radial, 0.2, 2.5)

        # --- Twist / Swirl (angular) ---
        # Max at center, fades towards radius
        ang = self.twist * strength * (1.0 - t)**2
        sinA, cosA = np.sin(ang), np.cos(ang)
        rx = (cosA * dx - sinA * dy)
        ry = (sinA * dx + cosA * dy)

        # --- Vertical "smile" stretch (center stronger) ---
        ry = ry * (1.0 + self.smile * strength * w_center)

        # Apply radial scaling
        rx *= radial
        ry *= radial

        # Final coordinates + add center back
        map_x = (rx + cx).astype(np.float32)
        map_y = (ry + cy).astype(np.float32)

        # Outside radius: identity
        outside = (t >= 1.0)
        map_x[outside] = xx[outside]
        map_y[outside] = yy[outside]
        return map_x, map_y

    def _feather_mask(self, h, w):
        """0..1 mask that fades at the box edges to hide seams."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = xx / (w - 1 + 1e-6)
        ny = yy / (h - 1 + 1e-6)
        fx = np.minimum(nx, 1 - nx)
        fy = np.minimum(ny, 1 - ny)
        edge = np.minimum(fx, fy)
        fw = self.feather * 0.5
        m = np.clip(edge / fw, 0, 1) ** 1.5
        return m.astype(np.float32)

    def _warp_roi(self, roi):
        h, w = roi.shape[:2]
        # Center around upper-middle of face (between eyes area)
        cx = w * 0.50
        cy = h * 0.42
        R  = self.radius_scale * min(w, h)

        # Build maps; strong values are handled smoothly
        map_x, map_y = self._maps_funhouse(h, w, cx, cy, R, self.strength)
        warped = cv2.remap(
            roi, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Feathered blend with original to avoid ROI edge artifacts
        mask = self._feather_mask(h, w)[..., None]
        out = (mask * warped.astype(np.float32) + (1 - mask) * roi.astype(np.float32)).astype(np.uint8)
        return out

    # -------- pipeline hook --------
    def apply(self, frame_bgr, frame_gray, faces):
        if not self.enabled or len(faces) == 0:
            return frame_bgr

        H, W = frame_bgr.shape[:2]
        # Track the largest face each frame (follows motion)
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # Pad ROI a bit so the deformation feels larger
        pad_x = int(0.15 * w)
        pad_y = int(0.12 * h)
        x = x - pad_x
        y = y - pad_y
        w = w + 2 * pad_x
        h = h + 2 * pad_y
        x, y, w, h = clamp_rect(x, y, w, h, W, H)

        roi = frame_bgr[y:y+h, x:x+w]
        warped = self._warp_roi(roi)

        out = frame_bgr.copy()
        out[y:y+h, x:x+w] = warped

        # HUD (optional)
        cv2.putText(out, f"Warp s={self.strength:.2f} tw={self.twist:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return out

    # Optional hotkeys (use from main.py if you like)
    def nudge(self, delta):
        self.strength = float(np.clip(self.strength + delta, 0.05, 0.95))
