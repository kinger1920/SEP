import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

CAT_ID = 15
DOG_ID = 16
OUTPUT_SIZE = (224, 224)
PAD_COLOR = (255, 255, 255)


model = YOLO()
model_seg = YOLO("yolov8n-seg.pt")


def box_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def letterbox_resize(crop, output_size=OUTPUT_SIZE, pad_color=PAD_COLOR):

    h, w = crop.shape[:2]
    target_w, target_h = output_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def crop_cat_or_dog(image, conf=0.25):

    if image is None:
        return None

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
    crop = letterbox_resize(crop, OUTPUT_SIZE)

    if is_pil:
        return Image.fromarray(crop)
    return crop


def crop_cat_or_dog_segmented(image, conf=0.25):

    if image is None:
        return None

    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = np.array(image)
    else:
        img_np = image

    results = model_seg.predict(
        source=image,
        classes=[CAT_ID, DOG_ID],
        conf=conf,
        retina_masks=True,
        verbose=False,
    )

    best = None

    for result in results:
        if result.masks is None or result.boxes is None:
            continue

        masks_data = result.masks.data

        for i in range(len(result.boxes)):
            mask_bool = masks_data[i].cpu().numpy() > 0.5
            area = int(mask_bool.sum())

            if best is None or area > best["area"]:
                xyxy = result.boxes[i].xyxy[0].tolist()
                best = {"mask": mask_bool, "box": xyxy, "area": area}

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


    masked_image = np.full_like(img_np, PAD_COLOR, dtype=np.uint8)
    mask_full = best["mask"]
    masked_image[mask_full] = img_np[mask_full]


    crop = masked_image[y1:y2, x1:x2]
    crop = letterbox_resize(crop, OUTPUT_SIZE)

    if is_pil:
        return Image.fromarray(crop)
    return crop