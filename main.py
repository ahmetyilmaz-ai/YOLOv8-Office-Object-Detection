from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / 'Dataset'
DATA_YAML = DATASET_DIR / 'data.yaml'
DEFAULT_MODEL = PROJECT_ROOT / 'yolov8n.pt'
RUNS_DIR = PROJECT_ROOT / 'runs' / 'detect'


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def find_default_image() -> Path:
    image_dir = DATASET_DIR / 'train' / 'images'
    ensure_exists(image_dir, 'Image directory')

    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No sample image found in: {image_dir}")


def find_latest_trained_weights() -> Path | None:
    if not RUNS_DIR.exists():
        return None

    candidates = sorted(RUNS_DIR.glob('train*/weights/best.pt'), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def train_model(epochs: int, imgsz: int) -> None:
    ensure_exists(DEFAULT_MODEL, 'Base model file')
    ensure_exists(DATA_YAML, 'Dataset YAML file')

    model = YOLO(str(DEFAULT_MODEL))
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device='cpu',
        workers=0,
    )
    print(f"Training finished. Output directory: {results.save_dir}")


def predict_image(source: Path | None, weights: Path | None, save: bool, show: bool) -> None:
    source = source or find_default_image()
    ensure_exists(source, 'Prediction image')

    chosen_weights = weights or find_latest_trained_weights() or DEFAULT_MODEL
    ensure_exists(chosen_weights, 'Model weights')

    model = YOLO(str(chosen_weights))
    results = model.predict(source=str(source), save=save, show=show, device='cpu')
    print(f"Prediction finished. Model used: {chosen_weights}")
    if save and results:
        print(f"Outputs saved to: {results[0].save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='YOLOv8 office object detection CLI')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')

    predict_parser = subparsers.add_parser('predict', help='Run prediction on an image')
    predict_parser.add_argument('--source', type=Path, default=None, help='Input image path')
    predict_parser.add_argument('--weights', type=Path, default=None, help='Path to .pt weights file')
    predict_parser.add_argument('--show', action='store_true', help='Show result in a window')
    predict_parser.add_argument('--no-save', action='store_true', help='Do not save result files')

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        args.command = 'train'
        args.epochs = 15
        args.imgsz = 640

    try:
        if args.command == 'train':
            train_model(epochs=args.epochs, imgsz=args.imgsz)
        elif args.command == 'predict':
            predict_image(source=args.source, weights=args.weights, save=not args.no_save, show=args.show)
        else:
            parser.error(f'Unknown command: {args.command}')
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
