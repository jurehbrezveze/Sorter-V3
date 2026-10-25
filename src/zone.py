import cv2
import json
import numpy as np


class Zone:
    def __init__(self, name, polygon, on_enter=None, on_exit=None, on_multiple=None):
        self.name = name
        self.polygon = np.array(polygon, np.int32)
        self.on_enter = on_enter or (lambda: None)
        self.on_exit = on_exit or (lambda: None)
        self.on_multiple = on_multiple or (lambda: None)
        self.frames = 5  # consecutive empty frames needed before exiting counts
        self.occupied = False
        self.multiple = False  # more than one brick in the zone
        self.boxes = []  # boxes of the bricks in the zone in the latest frame
        self.count = 0  # bricks in the zone in the latest frame
        self.streak = 0
        self.multiple_streak = 0

    def contains(self, x, y):
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def update(self, boxes):
        # A brick is in the zone when the centre of its box is inside the polygon
        self.boxes = [b for b in boxes if self.contains((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)]
        self.count = len(self.boxes)

        if self.count:
            self.streak = 0
            if not self.occupied:
                self.occupied = True
                self.on_enter()
        elif self.occupied:
            self.streak += 1
            if self.streak >= self.frames:
                self.occupied = False
                self.streak = 0
                self.on_exit()

        # multiple bricks in the zone: switching on or off both need `frames` frames in a row
        if (self.count > 1) == self.multiple:
            self.multiple_streak = 0
        else:
            self.multiple_streak += 1
            if self.multiple_streak >= self.frames:
                self.multiple = not self.multiple
                self.multiple_streak = 0
                if self.multiple:
                    self.on_multiple()

    def draw(self, frame):
        color = (0, 0, 255) if self.occupied else (200, 200, 200)
        cv2.polylines(frame, [self.polygon], True, color, 2)
        x, y = self.polygon[0]
        cv2.putText(frame, f"{self.name}: {self.count}",(int(x) + 4, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def load_polygons(path):
    return json.loads(path.read_text()) if path.exists() else {}
