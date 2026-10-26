import cv2
import serial
from pathlib import Path
import colorsys
import threading
import time
from ultralytics import YOLO
from zone import Zone, load_polygons
from APIs import brickognize, brick_image

model = YOLO(Path(__file__).parent.parent / "model/runs/segment/train/weights/best.pt")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
polygons = load_polygons(Path(__file__).parent / "zones.json")
pico = serial.Serial("COM3", 9600, timeout=0)

def pico_init():
    pico.write(b"v 0 80\n")
    pico.write(b"v 1 70\n")
    pico.write(b"r 0 1000\n")
    pico.write(b"s 1\n")
    pico.write(b"s 2\n")
    pico.write(b"p 0 1\n")

# Belt photo
PHOTO_DELAY = 0.5 # stop belt and wait for photo (sec)
PHOTO_MARGIN = 20  # extra space around the photo (px)
photo_at = None
reference = None

def lookup_part(part_path):
    global reference
    try:
        part_id = brickognize(part_path)
        print("Part id:", part_id)
        if part_id:
            reference = brick_image(part_id)
    except Exception as e:
        print("Part lookup failed:", e)


# C channel
def c_front_enter():
    if(zones[1].occupied):
        pico.write(b"b 1\n")
    
    pico.write(b"v 1 25\n")

    pico.write(b"b 0\n")
    pico.write(b"p 0 0\n")

def c_front_exit():
    pico.write(b"v 1 70\n")
    pico.write(b"r 0 1000\n")
    pico.write(b"s 1\n")
    pico.write(b"p 0 1\n")

# Belt
def belt_enter():
    if(zones[0].occupied):
        pico.write(b"b 1\n")

def belt_exit():
    pico.write(b"d 2 1\n")

    pico.write(b"s 1\n")

    if(not zones[0].occupied):
        pico.write(b"r 0 1000\n")
        pico.write(b"p 0 1\n")

def belt_multiple():
    pico.write(b"d 2 0\n")

def belt_front_enter():
    global photo_at
    pico.write(b"b 2\n")
    photo_at = time.time() + PHOTO_DELAY

def take_photo():
    if photo_zone.boxes:
        x1, y1, x2, y2 = photo_zone.boxes[0]
        # Clamp to the frame, a negative index would wrap around to the other side of the image
        height, width = clean_frame.shape[:2]
        x1, y1 = max(x1 - PHOTO_MARGIN, 0), max(y1 - PHOTO_MARGIN, 0)
        x2, y2 = min(x2 + PHOTO_MARGIN, width), min(y2 + PHOTO_MARGIN, height)
        part = clean_frame[y1:y2, x1:x2]
        part_path = Path(__file__).parent / "part.png"
        cv2.imwrite(str(part_path), part)
        cv2.imshow("Part", part)
        threading.Thread(target=lookup_part, args=(part_path,), daemon=True).start()
    else:
        print("No brick in photo zone")

    pico.write(b"s 2\n")

def belt_front_exit():
    return


photo_zone = Zone("r3", polygons["r3"], on_enter=belt_front_enter, on_exit=belt_front_exit)
zones = [
    Zone("r1", polygons["r1"], on_enter=c_front_enter, on_exit=c_front_exit),
    Zone("r2", polygons["r2"], on_enter=belt_enter, on_exit=belt_exit, on_multiple=belt_multiple),
    photo_zone,
]

pico_init()

while cv2.waitKey(1) & 0xFF != ord("q"):
    success, frame = camera.read()
    if not success:
        break

    clean_frame = frame.copy()  # without the drawn boxes, used for the part photo
    i = 0
    boxes =model.predict(frame, conf=0.25, verbose=False)[0].boxes.xyxy.int().tolist()

    for x1, y1, x2, y2 in boxes:
        hue = i / 10
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (r * 255, g * 255, b * 255), 2)
        i += 1

    for zone in zones:
        zone.update(boxes)
        zone.draw(frame)

    if photo_at is not None and time.time() >= photo_at:
        photo_at = None
        take_photo()

    if reference is not None:
        cv2.imshow("Reference", reference)
        reference = None

    cv2.imshow("BrickFinder", frame)


pico.write(b"b 0\n")
pico.write(b"b 1\n")
pico.write(b"b 2\n")
pico.write(b"p 0 0\n")

camera.release()
cv2.destroyAllWindows()
