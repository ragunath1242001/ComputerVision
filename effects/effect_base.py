class EffectBase:
    name = "base"
    enabled = True

    def toggle(self):
        self.enabled = not self.enabled

    def update_interaction(self, **kwargs):
        """Optional: receive external interaction controls (e.g., hand x position)."""
        pass

    def apply(self, frame_bgr, frame_gray, faces):
        """Return modified frame_bgr. Default: pass-through."""
        return frame_bgr
