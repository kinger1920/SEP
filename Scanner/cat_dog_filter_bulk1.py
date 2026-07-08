
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
#Nur Katze / Hund

CAT_ID = 15
DOG_ID = 16

# Model weights: swap for your own trained/fine-tuned weights if you have them
MODEL_PATH = "yolov8n.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_SIZE = (224, 224)


def box_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def get_single_biggest_detection(model: YOLO, image_path: str, conf: float = 0.25):

    results = model.predict(
        source=image_path,
        classes=[CAT_ID, DOG_ID],
        conf=conf,
        verbose=False,
    )

    best = None

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf.item())
            area = box_area(xyxy)
            label = "cat" if cls_id == CAT_ID else "dog"

            if best is None or area > best["area"]:
                best = {"label": label, "box": xyxy, "conf": confidence, "area": area}

    return best if best is not None else -1

#wenn man resize nicht will hier bearbeiten->
def crop_resize_and_save(image_path: str, detection, output_folder: Path, stem: str):

    if detection == -1:
        print(f"  [i] No cat or dog found, nothing saved for {stem}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"  [!] Could not read image, skipping: {image_path}")
        return

    h, w = img.shape[:2]
    x1, y1, x2, y2 = detection["box"]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        print(f"  [!] Invalid box, skipping: {stem}")
        return

    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    out_path = output_folder / f"{stem}_{detection['label']}.jpg"
    cv2.imwrite(str(out_path), resized)


def process_folder(input_folder: str, output_folder: str, conf: float = 0.25):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    image_paths = [
        p for p in sorted(input_folder.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    ]

    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_paths)} image(s). Processing...\n")

    for img_path in image_paths:
        print(f"- {img_path.name}")
        detection = get_single_biggest_detection(model, str(img_path), conf=conf)
        print(f"    biggest: {detection}")

        crop_resize_and_save(str(img_path), detection, output_folder, img_path.stem)

    print(f"\nDone. Cropped 224x224 images saved to: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python cat_dog_filter.py "<input_folder>" "<output_folder>"')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_folder(input_folder, output_folder)
# So this is all what I initially had. You could say what folder from the pictures shall be taken and where the results have to be placed. Worked great.
