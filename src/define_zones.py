import cv2
import json
import numpy as np
from pathlib import Path

ZONES_PATH = Path(__file__).parent / "zones.json"
SNAP = 10  # how close (px) a click must be to the first vertex to close the polygon

zones = json.loads(ZONES_PATH.read_text()) if ZONES_PATH.exists() else {}
points = []  # vertices of the polygon being drawn
mouse = (0, 0)


def near_first(x, y):
    return len(points) >= 3 and abs(x - points[0][0]) <= SNAP and abs(y - points[0][1]) <= SNAP


def on_mouse(event, x, y, flags, param):
    global mouse
    mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if near_first(x, y):
            n = 1
            while f"r{n}" in zones:
                n += 1
            zones[f"r{n}"] = points.copy()
            points.clear()
        else:
            points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Zones")
cv2.setMouseCallback("Zones", on_mouse)
print("Click = add vertex | click first vertex = close | Right click = remove vertex")
print("U = undo last zone | C = clear all | S = save | Q = quit")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("u") and zones:
        zones.popitem()
    if key == ord("c"):
        zones.clear()
        points.clear()
    if key == ord("s"):
        ZONES_PATH.write_text(json.dumps(zones, indent=2))
        print(f"Saved {len(zones)} zones to {ZONES_PATH}")

    success, frame = camera.read()
    if not success:
        break

    for name, polygon in zones.items():
        cv2.polylines(frame, [np.array(polygon, np.int32)], True, (255, 255, 0), 2)
        cv2.putText(frame, name, (polygon[0][0] + 4, polygon[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if points:
        cv2.polylines(frame, [np.array(points + [list(mouse)], np.int32)], False, (0, 255, 255), 1)
        for x, y in points:
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(points[0]), SNAP, (0, 255, 0) if near_first(*mouse) else (0, 255, 255), 1)

    cv2.imshow("Zones", frame)

camera.release()
cv2.destroyAllWindows()
