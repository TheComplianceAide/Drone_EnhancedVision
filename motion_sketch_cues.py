"""Brief highlights of changed pixels on the wide view, without object labels."""
import cv2
import numpy as np


class ChangeInk:
    def __init__(self):
        self.reset()

    def reset(self):
        self.image = None
        self.stamp = None

    def update(self, change, stamp, stable):
        if change is None or not stable:
            self.reset()
            return
        signal = change[:, :, 0]
        # Whole-scene changes are already visible in Motion Map. Drawing them
        # all over the source would obscure the operator's overview.
        strong = cv2.inRange(signal, 222, 255)
        if cv2.countNonZero(strong) > strong.size * .035:
            self.reset()
            return
        if self.image is None or self.stamp is None or stamp <= self.stamp:
            self.image = strong
        else:
            fade = min(255, round((stamp - self.stamp) * 255 / .8))
            self.image = cv2.max(cv2.subtract(self.image, fade), strong)
        self.stamp = stamp

    def draw(self, image):
        if self.image is None:
            return
        # Apply colour to the actual changed pixels, keeping the scene visible.
        ink = cv2.resize(self.image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        tint = cv2.merge((ink // 7, ink // 3, ink // 12))
        cv2.add(image, tint, dst=image)
