import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import torchaudio
except Exception as e:  # pragma: no cover
    torchaudio = None

try:
    from torchvision.io import read_video
except Exception:
    read_video = None


def find_videos(split_root: Path, categories: Sequence[str]) -> List[Tuple[Path, int]]:
    video_label_pairs: List[Tuple[Path, int]] = []
    category_to_index: Dict[str, int] = {c: i for i, c in enumerate(categories)}
    for category in categories:
        category_dir = split_root / category
        if not category_dir.exists():
            # Skip silently to allow partial datasets
            continue
        for ext in (".mp4", ".avi", ".mkv", ".mov"):
            for path in sorted(category_dir.rglob(f"*{ext}")):
                video_label_pairs.append((path, category_to_index[category]))
    return video_label_pairs


def uniform_sample_indices(num_frames: int, target: int) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames >= target:
        return [int(round(i * (num_frames - 1) / (target - 1))) for i in range(target)]
    # pad by repeating last frame
    idxs = list(range(num_frames))
    idxs.extend([num_frames - 1] * (target - num_frames))
    return idxs


def load_video_tensor(video_path: Path, num_frames: int = 150, size: int = 112) -> torch.Tensor:
    if read_video is None:
        raise RuntimeError("torchvision is required to read videos. Please install torchvision.")
    # read_video returns (video[T,H,W,C], audio[A,channels], info)
    vframes, _, _ = read_video(str(video_path), pts_unit="sec")
    if vframes.ndim != 4 or vframes.size(-1) != 3:
        raise RuntimeError(f"Unexpected video tensor shape for {video_path}: {tuple(vframes.shape)}")
    # Convert to float and CHW per frame
    vframes = vframes.float() / 255.0  # [T,H,W,3]
    t = vframes.shape[0]
    indices = uniform_sample_indices(t, num_frames)
    vframes = vframes[indices]  # [num_frames,H,W,3]
    # to [num_frames,3,H,W]
    vframes = vframes.permute(0, 3, 1, 2)
    # resize all frames to size x size using bilinear
    vframes = F.interpolate(vframes, size=(size, size), mode="bilinear", align_corners=False)
    # to (3, num_frames, size, size)
    vframes = vframes.permute(1, 0, 2, 3).contiguous()
    return vframes


def load_audio_spectrogram(video_path: Path, n_mels: int = 224, target_width: int = 763) -> torch.Tensor:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to build audio features. Please install torchaudio.")
    # Extract audio from video
    # Fallback: load audio track via torchaudio backend
    try:
        waveform, sample_rate = torchaudio.load(str(video_path))
    except Exception:
        # If direct load fails (common for videos), try ffmpeg backend
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "audio.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            waveform, sample_rate = torchaudio.load(str(wav_path))

    if waveform.ndim == 2 and waveform.size(0) > 1:
        # mixdown to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)  # [1, n_mels, time]
    spec = mel_db.squeeze(0)  # [n_mels, time]

    # Time pad/trim to target width
    time_len = spec.shape[-1]
    if time_len < target_width:
        pad = target_width - time_len
        spec = F.pad(spec, (0, pad), mode="constant", value=spec.min().item())
    elif time_len > target_width:
        start = (time_len - target_width) // 2
        spec = spec[:, start : start + target_width]
    return spec  # [n_mels, target_width]


def shard_list(items: List, num_shards: int) -> List[List]:
    if num_shards <= 0:
        return [items]
    shard_size = math.ceil(len(items) / num_shards)
    return [items[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]


def build_batches(
    raw_root: Path,
    out_root: Path,
    categories: Sequence[str],
    rng_seed: int = 1337,
) -> None:
    random.seed(rng_seed)
    out_train = out_root / "train"
    out_valid = out_root / "valid"
    out_test = out_root / "test"
    for d in (out_train, out_valid, out_test):
        d.mkdir(parents=True, exist_ok=True)

    # Discover videos
    train_pairs = find_videos(raw_root / "train", categories)
    valid_pairs = find_videos(raw_root / "valid", categories)
    test_pairs = find_videos(raw_root / "test", categories)

    # Shuffle for more even shards
    random.shuffle(train_pairs)
    random.shuffle(valid_pairs)
    random.shuffle(test_pairs)

    # Build .pkt (video-only) sets first
    def build_pkt(split_pairs: List[Tuple[Path, int]]) -> List[Tuple[torch.Tensor, int]]:
        pkt: List[Tuple[torch.Tensor, int]] = []
        for path, label in split_pairs:
            try:
                v = load_video_tensor(path)
                pkt.append((v, label))
            except Exception:
                continue
        return pkt

    train_pkt = build_pkt(train_pairs)
    valid_pkt = build_pkt(valid_pairs)
    test_pkt = build_pkt(test_pairs)

    # Save .pkt shards as expected by scripts
    # train: batch0..batch23.pkt (24 shards)
    for i, shard in enumerate(shard_list(train_pkt, 24)):
        if len(shard) == 0:
            continue
        torch.save(shard, out_train / f"batch{i}.pkt")
    # valid: only batch0.pkt used
    if len(valid_pkt) > 0:
        torch.save(valid_pkt, out_valid / "batch0.pkt")
    # test: only batch0.pkt used
    if len(test_pkt) > 0:
        torch.save(test_pkt, out_test / "batch0.pkt")

    # Build .pdt (video, audio_spectrogram, label)
    def build_pdt(split_pairs: List[Tuple[Path, int]]) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        pdt: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        for path, label in split_pairs:
            try:
                v = load_video_tensor(path)
                a = load_audio_spectrogram(path)
                pdt.append((v, a, label))
            except Exception:
                continue
        return pdt

    train_pdt = build_pdt(train_pairs)
    valid_pdt = build_pdt(valid_pairs)
    test_pdt = build_pdt(test_pairs)

    # Save .pdt shards as expected by scripts
    # train: batch_370..batch_3721.pdt (22 shards)
    for i, shard in enumerate(shard_list(train_pdt, 22)):
        if len(shard) == 0:
            continue
        torch.save(shard, out_train / f"batch_37{i}.pdt")

    # valid: batch_370.pdt and batch_371.pdt
    for i, shard in enumerate(shard_list(valid_pdt, 2)):
        if len(shard) == 0:
            continue
        torch.save(shard, out_valid / f"batch_37{i}.pdt")

    # test: batch_370.pdt, batch_371.pdt, batch_372.pdt
    for i, shard in enumerate(shard_list(test_pdt, 3)):
        if len(shard) == 0:
            continue
        torch.save(shard, out_test / f"batch_37{i}.pdt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kinetics-small .pkt/.pdt batches for MultiBench")
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Root directory containing raw videos under train/valid/test/<category>/*.mp4",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root for kinetics_small (e.g., /mnt/e/Laboratory/datasets/Kinetics400/kinetics_small)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=["archery", "breakdancing", "crying", "dining", "singing"],
        help="Category names (folder names) to include as the 5-class small split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    build_batches(raw_root, out_root, args.categories)


if __name__ == "__main__":
    main()


