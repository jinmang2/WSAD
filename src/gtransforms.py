from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_modes_mapping


class GroupResize:
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 256,
        resample: int = Image.BILINEAR,
    ):
        self.worker = transforms.Resize(size, interpolation=resample)

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:
        return [self.worker(img) for img in img_group]


class GroupTenCrop:
    def __init__(self, size: int = 224):
        self.worker = transforms.TenCrop(size)

    def __call__(self, img_group: List[Image.Image]) -> List[List[Image.Image]]:
        return [self.worker(img) for img in img_group]


class GroupToTensor:
    def __init__(self):
        pil_to_tensor = transforms.PILToTensor()
        self.worker = transforms.Lambda(
            lambda crops: torch.stack([pil_to_tensor(crop) for crop in crops])
        )

    def __call__(self, img_group: List[Sequence[Image.Image]]) -> List[torch.Tensor]:
        return [self.worker(img) for img in img_group]


class GroupStackTensor(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def __call__(self, tensor_group: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensor_group, dim=self.dim).float()


class RepeatTensor(nn.Module):
    def __init__(self, dim: int = 0, max_len: int = 16):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        repeats = [1] * tensor.dim()
        length = tensor.size(self.dim)
        repeats[self.dim] = (self.max_len + length - 1) // length
        return tensor.repeat(*repeats)[: self.max_len]


def get_ten_crop_transforms(
    bridge_type: str = "pillow",
    resize: Union[int, Tuple[int, int]] = 256,
    cropsize: Union[int, Tuple[int, int]] = 224,
    interpolation: Union[int, transforms.InterpolationMode] = 2,  # BILINEAR
    clip_first: bool = False,
    mean: Sequence[float] = [114.75, 114.75, 114.75],
    std: Sequence[float] = [57.375, 57.375, 57.375],
    clip_length: int = 16,
):
    if bridge_type == "pillow" and not isinstance(interpolation, int):
        interpolation = pil_modes_mapping(interpolation)

    dim = 0 if clip_first else 1

    if bridge_type == "pillow":
        transform = transforms.Compose(
            [
                GroupResize(size=resize, resample=interpolation),
                GroupTenCrop(size=cropsize),
                GroupToTensor(),
                GroupStackTensor(dim),
                transforms.Normalize(mean=mean, std=std),
                RepeatTensor(dim, clip_length),
            ]
        )
    elif bridge_type == "torch":
        transform = nn.Sequential(
            *[
                transforms.Resize(size=resize, interpolation=interpolation),
                transforms.TenCrop(size=cropsize),
                GroupStackTensor(1 - dim),
                transforms.Normalize(mean=mean, std=std),
                RepeatTensor(dim, clip_length),
            ]
        )

    return transform
