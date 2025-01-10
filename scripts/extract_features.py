import argparse
import os
from typing import Union

import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import decord
from src.dataset import TenCropVideoFrameDataset
from src.i3d import build_i3d_feature_extractor

DEFAULT_REPO_ID = "jinmang2/ucf_crime"
DEFAULT_DATASET_CONF_NAME = "anomaly"
DEFAULT_CACHE_DIR = "/content/drive/MyDrive/ucf_crime"


def load_ucf_crime_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str = DEFAULT_CACHE_DIR,
    config_name: str = DEFAULT_DATASET_CONF_NAME,
) -> datasets.DatasetDict:
    return load_dataset(repo_id, config_name, cache_dir=cache_dir)


def load_feature_extraction_model(model_name: str = "i3d_8x8_r50") -> torch.nn.Module:
    model = build_i3d_feature_extractor(model_name=model_name)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    device = next(model.parameters()).device
    return model, device


def extract_features(
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    model: torch.nn.Module,
    device: torch.device,
    outpath: str,
):
    if isinstance(dataset, datasets.DatasetDict):
        for mode, dset in dataset.items():
            new_outpath = os.path.join(outpath, mode)

            extract_features(dset, model, device, new_outpath)

        return None

    assert isinstance(dataset, datasets.Dataset), (
        "The type of dataset argument must be `datasets.Dataset` or"
        f"`datasets.DatasetDict`. Your input's type is {type(dataset)}."
    )

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    def _extract_features(video_dataset: torch.utils.data.Dataset) -> torch.Tensor:
        outputs = []
        dataloader = DataLoader(video_dataset, batch_size=16, shuffle=False)
        for _, inputs in enumerate(dataloader):
            # Unlike Tushar-N's B which is `n_videos`, our B is `n_clips`.
            # (B, 10, 16, 3, 224, 224) -> (B, 10, 3, 16, 224, 224)
            inputs = inputs.permute(0, 1, 3, 2, 4, 5)
            crops = []
            for crop_idx in range(inputs.shape[1]):
                crop = inputs[:, crop_idx].to(device)
                # (B, 3, 16, 224, 224) -> (B, 2048, 1, 1, 1)
                crop = model(crop)
                crops.append(crop.detach().cpu().numpy())
            outputs.append(crops)

        # stack
        _outputs = []
        for output in outputs:
            # [(B, 2048, 1, 1, 1)] * 10 -> (B, 10, 2048, 1, 1, 1)
            _outputs.append(np.stack(output, axis=1))
        # [(B, 10, 2048, 1, 1, 1)] * T -> (n_clips, 10, 2048, 1, 1, 1)
        # T = n_clips / B
        _outputs = np.vstack(_outputs)
        outputs = np.squeeze(_outputs)  # (n_clips, 10, 2048)

        return outputs

    for sample in tqdm(dataset):
        # check existence
        filename = sample["video_path"].split(os.sep)[-1].split(".")[0]
        savepath = os.path.join(outpath, filename + "_i3d.npy")

        if os.path.exists(savepath):
            continue

        # If the size of the video is larger than 1GB, divide it by the segment length.
        # After that, upload it to RAM and receive the tencrop result from the model.
        # Note That: The script below provides the result of converting bytes to killobytes.
        # https://huggingface.co/datasets/jinmang2/ucf_crime/blob/main/ucf_crime.py
        if sample["size"] > 1024**2:
            # The fps of the video in `ucf_crime` dataset is 30. Therefore, 3,000 video frames
            # are about 100 seconds long, which is a good video length for inference in colab pro+.
            # Among the transforms of `TenCropVideoFrameDataset`, `LoopPad` forcibly pads clips lower than
            # `frames_per_clip`(default: 16), so segment_length is designated as a multiple of 16.
            seg_len = 16 * 188  # 3,008
            # read video frames
            vr = decord.VideoReader(uri=sample["video_path"])
            segments = []
            for seg in tqdm(range(len(vr) // seg_len + 1)):
                seg_folder = os.path.join(outpath, filename)

                if not os.path.exists(seg_folder):
                    os.makedirs(seg_folder)

                seg_savepath = os.path.join(seg_folder, filename + f"_{seg}.npy")

                if os.path.exists(seg_savepath):
                    outputs = np.load(seg_savepath)
                else:
                    images = []
                    for i in [seg * seg_len + i for i in range(seg_len)]:
                        if i == len(vr):
                            break
                        arr = vr[i].asnumpy()
                        images.append(Image.fromarray(arr))
                    video_dataset = TenCropVideoFrameDataset(images)
                    # inference
                    outputs = _extract_features(video_dataset)
                    np.save(seg_savepath, outputs)

                segments.append(outputs)
            outputs = np.vstack(segments)
        else:
            # read video frames
            video_dataset = TenCropVideoFrameDataset(sample["video_path"])
            # inference
            outputs = _extract_features(video_dataset)

        # save
        np.save(savepath, outputs)


def segment_features(feature_path: str, seg_outpath: str, seg_length: int = 32):
    files = sorted(os.listdir(feature_path))
    for file in tqdm(files):
        if not file.endswith(".npy"):
            continue

        savepath = os.path.join(seg_outpath, file)
        if os.path.exists(savepath):
            continue

        # (nclips, 10, 2048) -> (10, nclips, 2048)
        features = np.load(os.path.join(feature_path, file)).transpose(1, 0, 2)

        divided_features = []
        for f in features:
            new_feat = np.zeros((seg_length, f.shape[1])).astype(np.float32)
            r = np.linspace(0, len(f), seg_length + 1, dtype=int)
            for i in range(seg_length):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(f[r[i] : r[i + 1], :], 0)
                else:
                    new_feat[i, :] = f[r[i], :]
            divided_features.append(new_feat)
        divided_features = np.array(divided_features, dtype=np.float32)

        np.save(savepath, divided_features)


def main(args: argparse.Namespace):
    feat_outpath = os.path.join(args.outdir, "anomaly_features")
    anomaly = load_ucf_crime_dataset(args.repo_id, args.cache_dir, args.config_name)
    model, device = load_feature_extraction_model(args.model_name)
    extract_features(anomaly, model, device, feat_outpath)

    seg_outpath = os.path.join(args.outdir, f"segment_features_{args.seg_length}")
    # Apply segments only for the train dataset
    segment_features(os.path.join(feat_outpath, "train"), seg_outpath, args.seg_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract video features and segment them."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="jinmang2/ucf_crime",
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/content/drive/MyDrive/ucf_crime",
        help="Cache directory for the dataset.",
    )
    parser.add_argument(
        "--config_name", type=str, default="anomaly", help="Dataset configuration name."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="i3d_8x8_r50",
        help="Feature extraction model name.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/content/drive/MyDrive/ucf_crime",
        help="Output directory for features.",
    )
    parser.add_argument(
        "--seg_length",
        type=int,
        default=32,
        help="Segment length for feature extraction.",
    )

    args = parser.parse_args()
    main(args)
