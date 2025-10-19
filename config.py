from dataclasses import dataclass

@dataclass
class AppConfig:
    camera_index: int = 0
    show_fps: bool = True
    # Face detector settings
    min_face_size: int = 120
    # Glasses overlay config
    glasses_path: str = "assets/overlays/glasses.png"
    glasses_scale: float = 1.1
    glasses_spacing: float = 1.00  # live-tweakable with [ and ]
    # Warp config
    warp_strength: float = 0.75  # 0..~0.4 bulge/stretch
    # Motion interaction (color tracker) HSV ranges for a bright object (e.g., blue cap)
    hsv_low: tuple = (90, 80, 80)
    hsv_high: tuple = (130, 255, 255)
