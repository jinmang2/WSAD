import importlib
import json
import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from .gtransforms import get_ten_crop_transforms

DEFAULT_FEATURE_HUB = "jinmang2/ucf_crime_tencrop_i3d_seg32"
DEFAULT_FILENAMES = {"train": "train.zip", "test": "test.zip"}


class BridgeType(Enum):
    PILLOW = "pillow"
    PYTORCH = "torch"


def is_decord_available():
    return importlib.util.find_spec("decord") is not None


def _build_feature_dataset(
    filepath: str, mode: str, dynamic_load: bool
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")

    zipf = zipfile.ZipFile(filepath)

    filenames = []
    values = {}
    for member in zipf.infolist():
        filename = member.filename.split("/")[-1]
        filenames.append(filename)
        value = np.load(zipf.open(member)) if not dynamic_load else member
        values[filename] = value

    if mode == "test":
        gt_path = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename="ground_truth.json",
            repo_type="dataset",
            force_download=True,
        )
        gt = json.load(open(gt_path))
        return FeatureDataset(
            filenames=filenames,
            values=values,
            labels=gt,
            open_func=zipf.open if dynamic_load else None,
        )

    normal_filenames = [fname for fname in filenames if "Normal" in fname]
    normal_kwargs = {
        "filenames": normal_filenames,
        "values": {fname: values[fname] for fname in normal_filenames},
        "open_func": zipf.open if dynamic_load else None,
    }
    abnormal_filenames = [fname for fname in filenames if "Normal" not in fname]
    abnormal_kwargs = {
        "filenames": abnormal_filenames,
        "values": {fname: values[fname] for fname in abnormal_filenames},
        "open_func": zipf.open if dynamic_load else None,
    }

    return {
        "normal": FeatureDataset(**normal_kwargs),
        "abnormal": FeatureDataset(**abnormal_kwargs),
    }


def build_feature_dataset(
    mode: str = "train",
    local_path: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    dynamic_load: bool = True,
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")
    assert sum([local_path is None, filename is None]) != 1

    if local_path is None:  # filename is also None
        filepath = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename=DEFAULT_FILENAMES[mode],
            cache_dir=cache_dir,
            revision=revision,
            repo_type="dataset",
        )
    else:  # local_path and filename aren't None
        filepath = os.path.join(local_path, filename)

    return _build_feature_dataset(filepath, mode, dynamic_load)


class FeatureDataset(Dataset):
    def __init__(
        self,
        filenames: List[str],
        values: Dict[str, Union[zipfile.ZipInfo, np.ndarray]],
        labels: Optional[Dict[str, float]] = None,
        open_func: Optional[Callable] = None,
    ):
        self.filenames = filenames
        self.values = values
        self.labels = labels

        self.open_func = open_func

    def __len__(self) -> int:
        return len(self.values)

    def open(self, value: Union[zipfile.ZipInfo, np.ndarray]) -> np.ndarray:
        if self.open_func is None:
            return value
        # dynamic loading
        return np.load(self.open_func(value))

    def add_magnitude(self, feature: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(feature, axis=2)[:, :, np.newaxis]
        feature = np.concatenate((feature, magnitude), axis=2)
        return feature

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fname = self.get_filename(idx)
        feature = self.open(self.values[fname])
        anomaly = 0.0 if "Normal" in fname else 1.0
        outputs = {
            "feature": self.add_magnitude(feature),
            "anomaly": np.array(anomaly, dtype=np.float32),
        }

        if self.labels is not None:
            label = np.array(self.labels[fname], dtype=np.float32)
            outputs.update({"label": label})

        return outputs

    def get_filename(self, idx: int) -> str:
        return self.filenames[idx]


class TencropVideoFrameDataset(Dataset):
    def __init__(
        self,
        video_path: Union[str, Path],
        bridge_type: Union[str, BridgeType] = "pillow",
        clip_length: int = 16,
        **transform_kwargs,
    ):
        super().__init__()
        if not isinstance(bridge_type, BridgeType):
            bridge_type = BridgeType(bridge_type)
        self.bridge_type = bridge_type

        if is_decord_available():
            import decord

            bridge = "torch" if bridge_type == BridgeType.PYTORCH else "native"
            decord.bridge.set_bridge(bridge)
        else:
            raise ImportError("To support decoding videos, please install `decord`.")

        self.video_reader = decord.VideoReader(uri=video_path)
        self.n_frames = len(self.video_reader)

        self.clip_length = clip_length
        self.transform = get_ten_crop_transforms(
            bridge_type=bridge_type.value,
            clip_length=clip_length,
            **transform_kwargs,
        )

    def __len__(self) -> int:
        return (self.n_frames - 1) // self.clip_length + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx, end_idx = idx * self.clip_length, (idx + 1) * self.clip_length
        indices = range(start_idx, min(self.n_frames, end_idx))
        clip = self.video_reader.get_batch(indices)
        if self.bridge_type == BridgeType.PILLOW:
            clip = list(map(Image.fromarray, clip.asnumpy()))
        tensor = self.transform(clip)
        return tensor
