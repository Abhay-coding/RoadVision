"""Check that the torchvision DeepLabV3 checkpoint can be loaded."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import RoadVisionSegmentor


if __name__ == "__main__":
    segmentor = RoadVisionSegmentor(architecture="deeplabv3", device="cpu")
    print(f"Loaded DeepLabV3 with {len(segmentor.class_names)} classes.")
