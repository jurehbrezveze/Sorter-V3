import cv2
import numpy as np
import requests


def brickognize(image_path):
    url = 'https://api.brickognize.com/predict/parts/'
    with open(image_path, 'rb') as image:
        files = {'query_image': ('part.png', image, 'image/png')}
        headers = {'accept': 'application/json'}
        x = requests.post(url, headers=headers, files=files, timeout=10)
    items = x.json().get("items", [])
    return items[0]["id"] if items else None


def brick_image(part_id):
    # BrickLink rejects requests without a browser User-Agent
    url = f"https://img.bricklink.com/ItemImage/PL/{part_id}.png"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    if response.status_code != 200:
        return None
    return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
