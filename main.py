import argparse
from code import interact
import cv2
import numpy as np

from config import AppConfig
from core.video import open_camera, FPS
from core.face_detecter import FaceDetector

from effects import FaceWarp, GlassesEffect, MotionInteraction

def parse_args():
    ap = argparse.ArgumentParser("IPCV Face Effects")
    ap.add_argument("--camera", type=int, default=None,
                    help="Webcam index (default from config)")
    ap.add_argument("--effects", type=str, default="warp,glasses,interaction",
                    help="Comma list: warp,glasses,interaction")
    ap.add_argument("--show-fps", type=int, default=None,
                    help="1 to show FPS, 0 to hide (default from config)")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---- Config (with safe fallbacks) ----
    cfg = AppConfig()
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.show_fps is not None:
        cfg.show_fps = bool(args.show_fps)

    cfg.min_face_size   = getattr(cfg, "min_face_size", 120)
    cfg.glasses_spacing = getattr(cfg, "glasses_spacing", 1.00)
    cfg.warp_strength   = getattr(cfg, "warp_strength", 0.60)
    cfg.hsv_low         = getattr(cfg, "hsv_low", (90, 80, 80))
    cfg.hsv_high        = getattr(cfg, "hsv_high", (130, 255, 255))

    # ---- Camera ----
    cap = open_camera(cfg.camera_index)
    if not cap or not cap.isOpened():
        print(f"Error: cannot open camera index {cfg.camera_index}")
        return

    # ---- Detector ----
    detector = FaceDetector(min_face=cfg.min_face_size, min_eye=20)

    # ---- Effects ----
    enabled = {k.strip().lower() for k in args.effects.split(",")}
    warp = FaceWarp(warp_strength=cfg.warp_strength)                 # strong warp impl
    glasses = GlassesEffect(detector, spacing=cfg.glasses_spacing)   # procedural (no PNG)
    interact = MotionInteraction(roi="right", min_area=2500)         # hand tracker

    warp.enabled = "warp" in enabled
    glasses.enabled = "glasses" in enabled
    interact.enabled = "interaction" in enabled

    fps = FPS()

    print("[Keys] 1:warp  2:glasses  3:interaction   [:spacing-  ]:spacing+   +/-:warp   q:quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection (used by warp and glasses)
        faces = detector.detect(gray)

        # 1) Interaction first (so we can drive others with its outputs)
        frame = interact.apply(frame, gray, faces)
        H, W = frame.shape[:2]
        tx, ty, fingers = interact.get_controls(W, H)

        # 2) Map hand position & gestures to effects  (YOUR UPDATED LOGIC)
        if tx is not None:
            warp.strength = 0.05 + 0.60 * tx        # left→soft, right→strong
        if ty is not None:
            glasses.spacing = max(0.6, min(1.6, 1.6 - ty))  # up→wider, down→narrower

        # Simple gestures
        if fingers >= 4:       # open palm
            glasses.enabled = True
        elif fingers == 0:     # fist/none
            glasses.enabled = False
        elif fingers in (2, 3):  # two/three fingers → quick warp boost
            warp.strength = min(0.9, warp.strength + 0.05)

        # 3) Apply other effects
        frame = warp.apply(frame, gray, faces)
        frame = glasses.apply(frame, gray, faces)

        # HUD + FPS
        if cfg.show_fps:
            f = fps.tick()
            cv2.putText(frame, f"FPS: {f:0.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        hud = []
        hud.append(f"[1] warp:    {'ON' if warp.enabled else 'off'}  s={warp.strength:.2f}")
        hud.append(f"[2] glasses: {'ON' if glasses.enabled else 'off'}  space={glasses.spacing:.2f}")
        hud.append(f"[3] interact: {'ON' if interact.enabled else 'off'}")
        y0 = 56
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, y0 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("IPCV Face Effects", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            warp.toggle()
        elif key == ord('2'):
            glasses.toggle()
        elif key == ord('3'):
            interact.toggle()
        elif key == ord('['):
            glasses.nudge_spacing(-0.04)
        elif key == ord(']'):
            glasses.nudge_spacing(+0.04)
        elif key in (ord('+'), ord('=')):
            warp.nudge(+0.05)
        elif key in (ord('-'), ord('_')):
            warp.nudge(-0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()