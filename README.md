# YOLOv8 Office Object Detection

This project uses YOLOv8 to detect office objects.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

### Train

```bash
python main.py train --epochs 15 --imgsz 640
```

Running `python main.py` also starts training by default.

### Predict

If a trained `best.pt` exists, the latest one is used automatically.
If not, the script falls back to `yolov8n.pt` and runs prediction on a sample dataset image.

```bash
python main.py predict
```

To predict on a specific image:

```bash
python main.py predict --source Dataset/train/images/example.jpg
```

## Fixed issues

- Missing `ultralytics` dependency caused startup failures.
- `Dataset/data.yaml` had an incorrect `path` value.
- `predict.py` referenced a non-existent test image.
- `main.py` now uses a safer CLI flow instead of training immediately on import.
