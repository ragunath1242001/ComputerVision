import numpy as np

def center_of_rect(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)

def order_two_points(p1, p2):
    return (p1, p2) if p1[0] <= p2[0] else (p2, p1)

def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return x, y, w, h

def lerp(a, b, t):
    return a + (b - a) * t

def map_range(val, a0, a1, b0, b1):
    t = (val - a0) / (a1 - a0 + 1e-6)
    return lerp(b0, b1, np.clip(t, 0.0, 1.0))
