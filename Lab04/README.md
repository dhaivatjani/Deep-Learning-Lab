# IT549: Deep Learning — Lab Assignment 4
## Object Detection Evolution: From R-CNN to YOLO

**Name:** Dhaivat Jani  
**ID:** 202511053  

---

## Project Overview

This lab traces the full evolution of object detection — from slow region-proposal pipelines (R-CNN, Fast R-CNN) to modern single-pass architectures (Faster R-CNN, YOLOv8). All tasks are implemented end-to-end on the **Fruit Images for Object Detection** dataset (Apple, Banana, Orange).

**Dataset:** [Fruit Images for Object Detection — Kaggle](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection)

---

## Repository Structure

```
Lab04/
├── Lab04.ipynb          # Main notebook with all tasks
└── README.md            # This file
```

---

## Tasks Completed

### Preparation — Ground Truth Visualization
- Loaded random training images and parsed Pascal VOC XML annotations
- Drew ground-truth bounding boxes with class labels using Matplotlib and OpenCV

---

### Task 1 — Intersection over Union (IoU)
Custom `compute_iou()` function implemented from scratch accepting `[x_min, y_min, x_max, y_max]` format.

Demonstrated on three box pairs:

| Case | Box A | Box B | IoU |
|---|---|---|---|
| Highly overlapping | [10,10,100,100] | [15,15,105,105] | ~0.81 |
| Partially overlapping | [0,0,50,50] | [30,30,80,80] | ~0.14 |
| Completely disjoint | [0,0,40,40] | [60,60,100,100] | 0.00 |

---

### Task 2 — Selective Search (R-CNN Step 1)
- Used OpenCV's `SelectiveSearchSegmentation` (fast mode)
- Extracted top 200 region proposals and visualized them overlaid on the original image

---

### Task 3 — R-CNN Bottleneck
- Loaded pretrained ResNet18 with final FC layer removed
- Looped through 100 region proposals, cropping and resizing each to 224×224 independently
- Passed each through the CNN to extract a 512-dim feature vector

**Result:**
```
R-CNN: 100 crops processed
Total time : 0.9540 seconds
Time/crop  : 9.54 ms
```

---

### Task 4 — Fast R-CNN (RoI Pooling)
- Passed the full image through the ResNet18 convolutional backbone **once**
- Spatial scale: `1/32` (224×224 → 7×7 feature map)
- Applied `torchvision.ops.roi_pool` on all 100 proposals simultaneously

**Result:**
```
Fast R-CNN: 100 RoIs processed
Total time : 0.0400 seconds  (40.0 ms)
```

**Speedup: 23.8× faster than R-CNN**

#### Conceptual Analysis
> In R-CNN, every region proposal is independently cropped, resized, and passed through the full CNN backbone — even when proposals share overlapping pixels. This leads to massive redundant computation. Fast R-CNN eliminates this by running the CNN **only once** on the entire image to produce a shared feature map, then extracting per-region features via RoI Pooling. The expensive convolutional computation is done exactly once regardless of proposal count.

---

### Task 5 — Faster R-CNN
- Loaded `fasterrcnn_resnet50_fpn` (pretrained on COCO)
- Ran inference and filtered predictions to confidence ≥ 0.80
- Visualized final filtered detections

#### Conceptual Analysis
> Selective Search is a hand-crafted, external algorithm that cannot be learned or jointly optimized. The **Region Proposal Network (RPN)** replaces it by sliding a small network over the **same shared feature map** already computed by the backbone — adding near-zero overhead. At each spatial location, it predicts objectness scores and box offsets for a set of anchor boxes (multiple scales and aspect ratios). Being fully differentiable, the RPN is trained end-to-end with the detector, learning to propose exactly the regions the classifier needs.

---

### Task 6 — Non-Maximum Suppression (NMS)
Custom `non_maximum_suppression()` implemented using the IoU function from Task 1.

**Algorithm:**
1. Sort boxes by confidence score (descending)
2. Select the highest-scoring box → keep as final prediction
3. Discard all remaining boxes with IoU > threshold against the selected box
4. Repeat until no boxes remain

#### Conceptual Analysis
> **High threshold (0.9):** NMS only suppresses boxes with near-total overlap. For tightly packed fruits, neighboring boxes overlap moderately but not at 90% — so most duplicates survive, producing many false positives (multiple boxes per apple).  
> **Low threshold (0.1):** Even slight overlap triggers suppression. Boxes for genuinely distinct nearby fruits get discarded, causing missed detections (one box where there should be several).  
> A moderate threshold (~0.4–0.5) balances these extremes for dense scenes.

---

### Task 7 — YOLOv8 Fine-Tuning

- Converted Pascal VOC XML annotations to YOLO format (normalized `x_center y_center width height`)
- Split dataset: 80% train / 10% val / 10% test
- Fine-tuned **YOLOv8n (Nano)** for 10 epochs on the fruit dataset

#### Test Set Evaluation (YOLOv8 Fine-tuned)

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **all** | 30 | 61 | 0.856 | 0.858 | **0.908** | **0.675** |
| apple | 9 | 13 | 1.000 | 0.913 | 0.982 | 0.823 |
| banana | 12 | 23 | 0.658 | 0.739 | 0.792 | 0.481 |
| orange | 14 | 25 | 0.909 | 0.920 | 0.950 | 0.720 |

---

## Model Comparison

| Model | Inference Time (ms) | Precision | Recall |
|---|---|---|---|
| Faster R-CNN (pretrained) | 72.1 | 0.4595 | 0.8361 |
| YOLOv8n (pretrained) | 19.3 | 0.5676 | 0.6885 |
| **YOLOv8n (fine-tuned)** | **13.4** | **0.8559** | **0.8575** |

**Key Observations:**
- Faster R-CNN achieves high recall but suffers from low precision and the slowest inference (72.1 ms), making it impractical for real-time use
- Pretrained YOLOv8n is fast but shows moderate accuracy since it was never trained on fruit-specific data
- Fine-tuned YOLOv8n delivers the best result across all three metrics — highest precision, competitive recall, and the fastest inference at 13.4 ms per image

---

## Environment

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.x |
| Torchvision | 0.x |
| Ultralytics | 8.3.x |
| OpenCV | 4.x |
| CUDA | RTX 5060 Laptop GPU |

---

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Lab04

# 2. Install dependencies
pip install torch torchvision ultralytics opencv-contrib-python matplotlib

# 3. Download dataset
kaggle datasets download -d mbkinaci/fruit-images-for-object-detection
unzip fruit-images-for-object-detection.zip

# 4. Open and run the notebook
jupyter notebook Lab04.ipynb
```

> Update the `DATASET_ROOT` path in the first code cell to point to your local dataset directory before running.
