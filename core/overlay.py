import cv2
import numpy as np

def alpha_blend(dst, src_bgra, x, y):
    """Overlay src_bgra (BGRA) onto dst (BGR) at (x,y) with alpha."""
    h, w = src_bgra.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    roi_dst = dst[y0:y1, x0:x1]
    roi_src = src_bgra[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    src_rgb = roi_src[..., :3]
    alpha = roi_src[..., 3:4].astype(np.float32) / 255.0
    roi_dst[:] = (alpha * src_rgb + (1 - alpha) * roi_dst).astype(np.uint8)

def resize_keep_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        scale = height / h
        return cv2.resize(img, (int(w*scale), height), interpolation=cv2.INTER_AREA)
    if height is None:
        scale = width / w
        return cv2.resize(img, (width, int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
