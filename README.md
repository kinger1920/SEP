# Fine-Grained Cat and Dog Breed Recognition

This repository contains the complete implementation for the Fine-Grained Animal Recognition Challenge, combining object localization with deep convolutional neural networks (CNNs) and confidence-based rejection gating to classify 20 cat and dog breeds alongside non-target image rejection (`-1`).

---

## System Architecture

The pipeline processes input images through a four-stage detect-then-classify architecture:

1. **Stage 1 (Object Localization)**: Uses a YOLO detector (`yolo26n.pt`) restricted to Cat (ID 15) and Dog (ID 16) COCO classes to filter out background clutter and non-animal objects.
2. **Stage 2 (Subject Extraction & Preprocessing)**: Applies a Maximum Surface Area Heuristic to isolate the primary animal, crops the region of interest (ROI), and resizes it to $224 \times 224$ pixels with ImageNet color normalization.
3. **Stage 3 (Breed Classification)**: Passes the normalized ROI crop through a Deep Convolutional Neural Network (custom 20-layer ConvNet with BatchNorm & Dropout or Fine-Tuned ResNet-34).
4. **Stage 4 (Confidence Rejection Gate)**: Converts output logits into Softmax probabilities. If top class confidence falls below $\tau = 0.20$, or if no animal is detected in Stage 1, the system outputs $-1$ (Reject).

---

## Environment Setup

### Prerequisites

- Python $\ge 3.8$
- PyTorch $\ge 2.0$ with CUDA or ROCm GPU acceleration support (optional, CPU supported)

### Dependency Installation

Install the required Python packages:

```bash
pip install torch torchvision opencv-python ultralytics pandas scikit-learn pillow tqdm
```

---

## Dataset & Preprocessing

The dataset is compiled from multiple open-source benchmarks (including Oxford-IIIT Pet, AtharvaTaras FCI Dog Breeds, and ma7555 Cat Breeds) and has been pre-filtered, weeded out, and balanced.

- Dataset Location: The complete, pre-cleaned dataset is pre-packaged and ready-to-use in the committed `images/` directory.
- Label Mapping: Target labels are defined in `images/labels.csv`:
  - `0` to `19`: Target cat and dog breeds (10 cat species, 10 dog species).
  - `-1`: Non-animal confounders and background images (rejection class).

No additional data downloading or manual preprocessing is required—the committed `images/` directory contains all data required for evaluation.

---

## Explainable AI (xAI) Scripts

To inspect model interpretability, analyze spatial focus, and verify that predictions are driven by animal features rather than background clutter, the repository includes three visual explanation tools.

To evaluate a specific input image or inspect different convolutional layers, update the `test_image_path` variable and `target_layers` dictionary directly inside `gradcam.py`, `occlusion.py`, or `layer_activations.py` before executing the script.

### 1. Grad-CAM (`gradcam.py`)

Computes Gradient-Weighted Class Activation Maps by extracting backpropagated gradients at intermediate and late convolutional layers.

- Functionality Multiplies forward feature maps by gradient channel importance weights to generate heatmaps highlighting discriminative region focus.
- Execution:
  ```bash
  python gradcam.py
  ```
- Output: Generates overlay heatmaps saved as `gradcam_<breed>_<layer>_result.jpg`.

### 2. Occlusion Sensitivity (`occlusion.py`)

Measures output probability drop by systematically obscuring input image regions using a sliding square masking patch (`patch_size=32`, `stride=16`).

- Functionality: High probability drop (warm colors: red/yellow) indicates critical feature regions (e.g., animal face, ears); low drop (cool colors: blue) indicates irrelevant regions.
- Execution:
  ```bash
  python occlusion.py
  ```
- Output: Saves spatial sensitivity heatmaps as `occlusion_<breed>_result.jpg`.

### 3. Layer Activations (`layer_activations.py`)

Extracts raw intermediate feature activations across early, middle, and late convolutional layers using PyTorch forward hooks.

- Functionality: Computes the mean spatial activation map across feature channels to track how visual representations evolve from low-level edges to high-level semantic body parts.
- Execution
  ```bash
  python layer_activations.py
  ```
- Output: Saves normalized activation maps as `activation_<breed>_<layer>_result.jpg`.

---

## Running Inference & Evaluation

To evaluate the complete model pipeline on the dataset, execute `inference.py` passing the target image folder:

```bash
python inference.py --image-folder images
```

If evaluating on a separate held-out validation folder:

```bash
python inference.py --image-folder validation
```

### Expected Output

`inference.py` evaluates every image in the folder and prints standard evaluation metrics:

- Overall Classification Accuracy
- Macro-averaged Precision, Recall, and F1-Score
- Class-wise Precision, Recall, and F1-Score for each of the 20 breeds and the rejection class (`-1`)
- Full $21 \times 21$ Confusion Matrix (rows = true class, columns = predicted class)
