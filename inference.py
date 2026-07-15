"""Evaluation harness for the Fine-grained Animal Recognition project.

We run this script on the held-out test set, so do not change the interface.
Implement your solution as the `Model` below: an `nn.Module` whose `forward`
takes a PIL image and returns a predicted class index, an integer in
{-1, 0, ..., 19}, where -1 means "reject", i.e. no target species is present.
Inside `forward` you are free to do anything you like: run an off-the-shelf
detector, find bounding boxes, crop the largest animal, classify the crop,
decide when to return -1, and so on.

The script reads `labels.csv` from the image folder, with columns
`filename,label`, where `label` is the integer class index from CLASSES (or -1
for confounders / images with no target species). The images themselves are a
flat, numbered set (0001.jpg, 0002.jpg, ...) sitting next to `labels.csv`. The
script runs your model on every image and prints the standard classification
metrics.

    python inference.py --image-folder <folder>
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",         #  0
    "Bengal",             #  1
    "Birman",             #  2
    "Bombay",             #  3
    "British_Shorthair",  #  4
    "Maine_Coon",         #  5
    "Ragdoll",            #  6
    "Sphynx",             #  7
    "Tabby",              #  8
    "Tiger_Cat",          #  9
    "Beagle",             # 10
    "Pug",                # 11
    "Boxer",              # 12
    "Shiba_Inu",          # 13
    "Samoyed",            # 14
    "Golden_Retriever",   # 15
    "German_Shepherd",    # 16
    "Siberian_Husky",     # 17
    "Dalmatian",          # 18
    "Rottweiler",         # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    """TODO (students): replace this with your own model.

    Contract: given a PIL image, return a class index in {-1, 0, ..., 19}.
    The placeholder below is a uniform random guesser so the script runs.
    """

    def __init__(self):
        from src.model import ConvolutionalNetwork
        import torchvision.transforms as transforms
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.classifier = TransferLearningNetwork(num_classes=20, pretrained=True, freeze_backbone=False).to(self.device)
        self.classifier = ConvolutionalNetwork().to(self.device)
        state_dict = torch.load("src/model_v4.pth", map_location=self.device)
        self.classifier.load_state_dict(state_dict)

        # self.detector = YOLO("src/Scanner/yolo26n.pt")

        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def forward(self, image: Image.Image) -> int:
        import torch
        from src.Scanner.cat_dog_filter1 import crop_cat_or_dog

        # STUFE 1 & 2: YOLO Detektion & Zuschneiden
        cropped_image = crop_cat_or_dog(image)
        if cropped_image is None:
            return -1

        # STUFE 3: Klassifikation
        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        # STUFE 4: Threshold
        if confidence.item() < 0.20:
            return -1

        return int(prediction.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="validation")
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model().eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred = model(image)
            y_true.append(int(label))
            y_pred.append(int(pred))

    labels = [REJECT] + list(range(NUM_CLASSES))
    target_names = ["reject(-1)"] + CLASSES
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, digits=3,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
