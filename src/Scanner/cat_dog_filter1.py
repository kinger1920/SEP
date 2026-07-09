# How to use this:
"""
img = cv2.imread("image.jpg")

cropped = crop_cat_or_dog(img)

if cropped is not None:
    # cropped is already 224x224
    cv2.imshow("Crop", cropped)
"""
# Now what I made cause we need a Function :: Image -> Cropped Image
import cv2
from ultralytics import YOLO

CAT_ID = 15
DOG_ID = 16
OUTPUT_SIZE = (224, 224)

# Load the model once
model = YOLO()


def box_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def crop_cat_or_dog(image, conf=0.25):
    """
    Input:
        image (OpenCV image / numpy array OR PIL Image)

    Returns:
        224x224 cropped image of the biggest cat/dog (same type as input),
        or None if no cat/dog is found.
    """
    if image is None:
        return None

    import numpy as np
    from PIL import Image

    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image)
    else:
        img_np = image

    results = model.predict(
        source=image,
        classes=[CAT_ID, DOG_ID],
        conf=conf,
        verbose=False,
    )

    best = None

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            area = box_area(xyxy)

            if best is None or area > best["area"]:
                best = {"box": xyxy, "area": area}

    if best is None:
        return None

    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = map(int, best["box"])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img_np[y1:y2, x1:x2]
    crop = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    if is_pil:
        return Image.fromarray(crop)
    return crop


