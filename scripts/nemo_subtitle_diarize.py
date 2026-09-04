#!/usr/bin/env python3
"""Exact-row subtitle diarization with NVIDIA NeMo cascaded components.

This script keeps subtitle JSON rows as the final diarization decision units,
while using row-centered multiscale audio windows for speaker evidence. The
full path uses TitaNet embeddings, NeMo spectral clustering, optional MSDD
local pairwise refinement, and a deterministic adjacent same-color constraint.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - full NeMo path requires numpy.
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - pure helper tests do not require torch.
    torch = None


DEFAULT_JSON = Path("data/labels/American_Fiction_2023_with_colors.json")
DEFAULT_VIDEO = Path("data/movies/American_Fiction_2023.mp4")
DEFAULT_EMBEDDING_DIR = Path("data/embeddings/American_Fiction_2023_nemo_exact")
DEFAULT_OUTPUT_DIR = Path("data/clusters/American_Fiction_2023_nemo_exact")
DEFAULT_OUTPUT_RTTM = Path("data/clusters/American_Fiction_2023_nemo_exact.rttm")
DEFAULT_SCALE_WINDOWS = "3.0,2.0,1.5,1.0,0.5"
DEFAULT_RTTM_FILE_ID = "American_Fiction_2023"

TAG_RE = re.compile(r"<[^>]+>")
VTT_TIMING_RE = re.compile(r"^\s*(\d\d:\d\d:\d\d(?:\.\d+)?)\s+-->\s+(\d\d:\d\d:\d\d(?:\.\d+)?)")
VTT_SPAN_RE = re.compile(r"<c(?:\.([A-Za-z0-9_-]+))?[^>]*>(.*?)</c>", re.DOTALL)
NON_ACTIONABLE_COLOR_KEYS = {"", "unknown", "none", "null", "nan", "n/a", "na"}


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the full NeMo diarization pipeline. "
            "Install the project venv dependencies before running model inference."
        )
    return torch


def require_numpy():
    if np is None:
        raise RuntimeError(
            "NumPy is required for the full NeMo diarization pipeline. "
            "Install the project venv dependencies before running model inference."
        )
    return np


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressLogger:
    def __init__(self, enabled: bool = True, stream: Any | None = None) -> None:
        self.enabled = enabled
        self.started_at = time.monotonic()
        self.stream = stream

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = format_elapsed(time.monotonic() - self.started_at)
        print(f"[progress {elapsed}] {message}", flush=True, file=self.stream)


@dataclass(frozen=True)
class SubtitleRow:
    row_index: int
    start: float
    end: float
    color: str
    text: str
    gold_speaker: str | None = None
    color_cue_confidence: float | None = None
    color_cue_ambiguous: bool | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def source_unit(self) -> dict[str, Any]:
        return {
            "unit_id": self.row_index,
            "row_index": self.row_index,
            "cue_index": self.row_index,
            "span_index": 1,
            "source_type": "json",
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration": round(self.duration, 6),
            "color": self.color,
            "text": self.text,
            "gold_speaker": self.gold_speaker,
            "color_cue_confidence": self.color_cue_confidence,
            "color_cue_ambiguous": self.color_cue_ambiguous,
        }


@dataclass(frozen=True)
class RowScaleWindow:
    row_index: int
    scale_index: int
    scale_window: float
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact-subtitle-row NeMo cascaded diarization with TitaNet "
            "embeddings, NeMo clustering, optional MSDD, and color constraints."
        )
    )
    parser.add_argument(
        "--subtitle",
        type=Path,
        default=DEFAULT_JSON,
        help="Input subtitle JSON or VTT file with subtitle rows and optional color cues.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input movie/video file used for audio extraction.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-rttm",
        type=Path,
        default=DEFAULT_OUTPUT_RTTM,
        help="Final predicted diarization RTTM file to write.",
    )
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--speaker-model", default="titanet_large")
    parser.add_argument("--msdd-model", default="diar_msdd_telephonic")
    parser.add_argument("--max-num-speakers", type=int, default=80)
    parser.add_argument("--num-speakers", type=parse_optional_int, default=None)
    parser.add_argument(
        "--neural-refinement",
        choices=("off", "local-pairwise"),
        default="local-pairwise",
        help="Use MSDD local pairwise refinement or keep NeMo clustering labels.",
    )
    parser.add_argument(
        "--scale-windows",
        default=DEFAULT_SCALE_WINDOWS,
        help="Comma-separated multiscale window lengths in descending order.",
    )
    parser.add_argument("--min-window", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--audio-cache", type=Path)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--rttm-file-id", default=DEFAULT_RTTM_FILE_ID)
    parser.add_argument("--max-rows", type=int, help="Process only the first N rows.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help=(
            "Print inner-loop progress every N embedding batches or MSDD pairs. "
            "Use 0 to keep only coarse stage messages."
        ),
    )
    parser.add_argument("--max-rp-threshold", type=float, default=0.25)
    parser.add_argument("--sparse-search-volume", type=int, default=30)
    parser.add_argument("--enhanced-count-thres", type=int, default=80)
    parser.add_argument("--chunk-cluster-count", type=int, default=50)
    parser.add_argument("--embeddings-per-chunk", type=int, default=10000)
    parser.add_argument("--msdd-block-size", type=int, default=64)
    parser.add_argument("--msdd-block-overlap", type=int, default=16)
    parser.add_argument("--msdd-max-pairs-per-block", type=int, default=12)
    parser.add_argument("--msdd-relabel-threshold", type=float, default=0.70)
    parser.add_argument("--msdd-relabel-margin", type=float, default=0.15)
    parser.add_argument(
        "--disable-color-run-merge",
        action="store_true",
        help=(
            "Disable the post-processing rule that forces adjacent subtitle rows "
            "with the same usable color cue to share one speaker."
        ),
    )
    return parser.parse_args()


def seconds_from_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def plain_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_scale_windows(value: str) -> list[float]:
    windows = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not windows:
        raise ValueError("--scale-windows must contain at least one value")
    if any(window <= 0 for window in windows):
        raise ValueError("--scale-windows values must be positive")
    if windows != sorted(windows, reverse=True):
        raise ValueError("--scale-windows must be in descending order for NeMo multiscale use")
    return windows


def cue_blocks(vtt_text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in vtt_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_vtt_cue_color(cue_text: str) -> tuple[str, float, bool]:
    color_lengths: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, match in enumerate(VTT_SPAN_RE.finditer(cue_text)):
        color = (match.group(1) or "unknown").strip().lower()
        text = plain_text(match.group(2))
        if not text:
            continue
        color_lengths[color] += len(text)
        first_seen.setdefault(color, index)

    if not color_lengths:
        return "unknown", 0.0, True

    ranked = sorted(color_lengths.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
    top_color, top_length = ranked[0]
    total_length = sum(color_lengths.values())
    confidence = top_length / total_length if total_length > 0 else 0.0
    ambiguous = len(ranked) > 1 and confidence < 0.8
    return top_color, round(confidence, 6), ambiguous


def parse_vtt_subtitle_rows(vtt_path: Path, max_rows: int | None = None) -> list[SubtitleRow]:
    parsed: list[SubtitleRow] = []
    for block in cue_blocks(vtt_path.read_text(encoding="utf-8-sig")):
        timing_index = next((index for index, line in enumerate(block) if "-->" in line), None)
        if timing_index is None:
            continue
        match = VTT_TIMING_RE.match(block[timing_index])
        if match is None:
            continue
        start = seconds_from_timestamp(match.group(1))
        end = seconds_from_timestamp(match.group(2))
        if end <= start:
            continue
        cue_text = "\n".join(block[timing_index + 1 :])
        text = plain_text(cue_text)
        color, confidence, ambiguous = parse_vtt_cue_color(cue_text)
        parsed.append(
            SubtitleRow(
                row_index=len(parsed) + 1,
                start=start,
                end=end,
                color=color,
                text=text,
                color_cue_confidence=confidence,
                color_cue_ambiguous=ambiguous,
            )
        )
        if max_rows is not None and len(parsed) >= max_rows:
            break
    return parsed


def parse_json_subtitle_rows(json_path: Path, max_rows: int | None = None) -> list[SubtitleRow]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = data.get("mappings_compact")
    if not isinstance(rows, list):
        raise ValueError(f"Expected {json_path} to contain key 'mappings_compact'")

    parsed: list[SubtitleRow] = []
    for row_index, row in enumerate(rows, start=1):
        start = seconds_from_timestamp(str(row["start_time"]))
        end = seconds_from_timestamp(str(row["end_time"]))
        if end <= start:
            continue
        text = row.get("vtt_subtitle") or row.get("moviesum_subtitle") or ""
        parsed.append(
            SubtitleRow(
                row_index=row_index,
                start=start,
                end=end,
                color=str(row.get("color_cue") or "unknown"),
                text=plain_text(str(text)),
                gold_speaker=row.get("speaker"),
                color_cue_confidence=row.get("color_cue_confidence"),
                color_cue_ambiguous=row.get("color_cue_ambiguous"),
            )
        )
        if max_rows is not None and len(parsed) >= max_rows:
            break
    return parsed


def parse_subtitle_rows(subtitle_path: Path, max_rows: int | None = None) -> list[SubtitleRow]:
    suffix = subtitle_path.suffix.lower()
    if suffix in {".vtt", ".webvtt"}:
        return parse_vtt_subtitle_rows(subtitle_path, max_rows=max_rows)
    return parse_json_subtitle_rows(subtitle_path, max_rows=max_rows)


def require_program(program: str) -> None:
    if shutil.which(program) is None:
        raise RuntimeError(f"Required executable not found on PATH: {program}")


def extract_audio(
    video_path: Path,
    audio_path: Path,
    sample_rate: int,
    overwrite: bool,
    progress: ProgressLogger | None = None,
) -> None:
    require_program("ffmpeg")
    if audio_path.exists() and not overwrite:
        if progress is not None:
            progress.log(f"audio cache found: {audio_path}")
        return

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.log(f"extracting {sample_rate} Hz mono audio: {video_path} -> {audio_path}")
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(audio_path),
    ]
    subprocess.run(command, check=True)
    if progress is not None:
        progress.log(f"audio extraction complete: {audio_path}")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def choose_device(requested: str) -> str:
    torch_mod = require_torch()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch_mod.cuda.is_available() else "cpu"


def build_scale_windows(
    rows: list[SubtitleRow],
    scale_windows: list[float],
    audio_duration: float,
    min_window: float,
) -> dict[int, list[RowScaleWindow]]:
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")
    if min_window <= 0:
        raise ValueError("min_window must be positive")

    by_scale: dict[int, list[RowScaleWindow]] = {}
    for scale_index, scale_window in enumerate(scale_windows):
        windows = []
        for row in rows:
            target_duration = min(max(row.duration, scale_window, min_window), audio_duration)
            start = row.midpoint - target_duration / 2.0
            end = row.midpoint + target_duration / 2.0
            if start < 0:
                end -= start
                start = 0.0
            if end > audio_duration:
                start -= end - audio_duration
                end = audio_duration
                start = max(0.0, start)
            if end <= start:
                start = max(0.0, min(row.start, audio_duration - min_window))
                end = min(audio_duration, start + min_window)
            windows.append(
                RowScaleWindow(
                    row_index=row.row_index,
                    scale_index=scale_index,
                    scale_window=scale_window,
                    start=round(start, 6),
                    end=round(end, 6),
                )
            )
        by_scale[scale_index] = windows
    return by_scale


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_segments_metadata(
    rows: list[SubtitleRow],
    windows_by_scale: dict[int, list[RowScaleWindow]],
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "row_index": row.row_index,
                "start": round(row.start, 6),
                "end": round(row.end, 6),
                "duration": round(row.duration, 6),
                "color": row.color,
                "text": row.text,
                "gold_speaker": row.gold_speaker,
                "color_cue_confidence": row.color_cue_confidence,
                "color_cue_ambiguous": row.color_cue_ambiguous,
                "embedding_windows": [
                    {
                        "scale_index": scale_index,
                        "scale_window": window.scale_window,
                        "start": window.start,
                        "end": window.end,
                        "duration": round(window.duration, 6),
                    }
                    for scale_index, windows in sorted(windows_by_scale.items())
                    for window in [windows[index]]
                ],
            }
        )
    return output


def write_nemo_manifest(
    path: Path,
    audio_path: Path,
    windows: list[RowScaleWindow],
    file_id: str,
) -> None:
    rows = []
    resolved_audio = str(audio_path.resolve())
    for window in windows:
        rows.append(
            {
                "audio_filepath": resolved_audio,
                "offset": round(window.start, 6),
                "duration": round(window.duration, 6),
                "label": "infer",
                "text": "-",
                "uniq_id": f"{file_id}_row_{window.row_index:06d}_scale_{window.scale_index}",
                "row_index": window.row_index,
                "scale_index": window.scale_index,
            }
        )
    write_jsonl(path, rows)


def load_speaker_model(model_path: str, device: str):
    torch_mod = require_torch()
    try:
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR is required for embedding extraction. Install it with "
            "`pip install 'nemo_toolkit[asr]'` in a compatible environment."
        ) from exc

    map_location = torch_mod.device(device)
    if model_path.endswith(".nemo"):
        model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=map_location)
    elif model_path.endswith(".ckpt"):
        model = EncDecSpeakerLabelModel.load_from_checkpoint(model_path, map_location=map_location)
    else:
        model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=model_path,
            map_location=map_location,
        )
    model = model.to(map_location)
    model.eval()
    return model


def extract_embeddings_for_manifest(
    model: Any,
    manifest_path: Path,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    expected_count: int,
    scale_number: int,
    scale_count: int,
    progress_every: int,
    progress: ProgressLogger | None = None,
) -> Any:
    torch_mod = require_torch()
    data_config = {
        "manifest_filepath": str(manifest_path),
        "sample_rate": sample_rate,
        "batch_size": batch_size,
        "trim_silence": False,
        "labels": None,
        "num_workers": num_workers,
    }
    if progress is not None:
        progress.log(
            f"scale {scale_number}/{scale_count}: setting up manifest "
            f"{manifest_path} ({expected_count} rows)"
        )
    model.setup_test_data(data_config)
    dataloader = model.test_dataloader()
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    batches = []
    rows_done = 0
    with torch_mod.no_grad():
        for batch_index, test_batch in enumerate(dataloader, start=1):
            moved = [item.to(model.device) if hasattr(item, "to") else item for item in test_batch]
            audio_signal = moved[0]
            audio_signal_len = moved[1]
            try:
                _, embeddings = model.forward(
                    input_signal=audio_signal,
                    input_signal_length=audio_signal_len,
                )
            except TypeError:
                _, embeddings = model.forward(input_signal=audio_signal, length=audio_signal_len)
            embeddings = embeddings.detach().float().cpu().reshape(-1, embeddings.shape[-1])
            batches.append(embeddings)
            rows_done += embeddings.shape[0]
            should_log = (
                progress is not None
                and progress_every > 0
                and (
                    batch_index == 1
                    or batch_index % progress_every == 0
                    or rows_done >= expected_count
                )
            )
            if should_log:
                total_text = str(total_batches) if total_batches is not None else "?"
                progress.log(
                    f"scale {scale_number}/{scale_count}: embedded batch "
                    f"{batch_index}/{total_text}, rows {min(rows_done, expected_count)}/"
                    f"{expected_count}"
                )

    if not batches:
        raise RuntimeError(f"No embeddings were extracted for {manifest_path}")
    output = torch_mod.cat(batches, dim=0)
    if output.shape[0] != expected_count:
        raise ValueError(
            f"Embedding row count mismatch for {manifest_path}: "
            f"expected {expected_count}, got {output.shape[0]}"
        )
    output = l2_normalize_torch(output)
    if progress is not None:
        progress.log(
            f"scale {scale_number}/{scale_count}: embeddings complete "
            f"({output.shape[0]} rows x {output.shape[1]} dims)"
        )
    return output


def extract_multiscale_embeddings(
    rows: list[SubtitleRow],
    windows_by_scale: dict[int, list[RowScaleWindow]],
    audio_path: Path,
    embedding_dir: Path,
    speaker_model: str,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    device: str,
    file_id: str,
    progress_every: int,
    progress: ProgressLogger | None = None,
) -> dict[int, Any]:
    manifests_dir = embedding_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.log(f"loading speaker embedding model '{speaker_model}' on {device}")
    model = load_speaker_model(speaker_model, device)
    if progress is not None:
        progress.log(f"speaker embedding model ready: {speaker_model}")

    scale_embeddings: dict[int, Any] = {}
    scale_count = len(windows_by_scale)
    for scale_index, windows in sorted(windows_by_scale.items()):
        manifest_path = manifests_dir / f"scale_{scale_index:02d}.json"
        write_nemo_manifest(manifest_path, audio_path, windows, file_id)
        if progress is not None:
            durations = [window.duration for window in windows]
            progress.log(
                f"scale {scale_index + 1}/{scale_count}: wrote manifest with "
                f"{len(windows)} windows, duration range "
                f"{min(durations):.3f}-{max(durations):.3f}s"
            )
        scale_embeddings[scale_index] = extract_embeddings_for_manifest(
            model=model,
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            expected_count=len(rows),
            scale_number=scale_index + 1,
            scale_count=scale_count,
            progress_every=progress_every,
            progress=progress,
        )
        if progress is not None:
            progress.log(f"finished scale {scale_index + 1}/{scale_count}")
    return scale_embeddings


def l2_normalize_torch(matrix: Any) -> Any:
    torch_mod = require_torch()
    matrix = matrix.float()
    if matrix.ndim == 1:
        norm = torch_mod.linalg.vector_norm(matrix)
        return matrix / norm if norm > 0 else matrix
    norms = torch_mod.linalg.vector_norm(matrix, dim=1, keepdim=True)
    norms = torch_mod.where(norms > 0, norms, torch_mod.ones_like(norms))
    return matrix / norms


def save_embedding_package(
    path: Path,
    rows: list[SubtitleRow],
    windows_by_scale: dict[int, list[RowScaleWindow]],
    scale_embeddings: dict[int, Any],
    scale_windows: list[float],
    speaker_model: str,
    audio_path: Path,
) -> None:
    torch_mod = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = torch_mod.stack([scale_embeddings[index] for index in sorted(scale_embeddings)])
    torch_mod.save(
        {
            "embeddings": stacked,
            "rows": row_segments_metadata(rows, windows_by_scale),
            "scale_windows": scale_windows,
            "speaker_model": speaker_model,
            "audio_cache": str(audio_path),
            "base_scale_index": len(scale_windows) - 1,
        },
        path,
    )


def build_clustering_inputs(
    file_id: str,
    rows: list[SubtitleRow],
    windows_by_scale: dict[int, list[RowScaleWindow]],
    scale_embeddings: dict[int, Any],
    scale_windows: list[float],
) -> dict[str, dict[str, Any]]:
    torch_mod = require_torch()
    embedding_parts = []
    timestamp_parts = []
    segment_counts = []
    base_scale_index = len(scale_windows) - 1

    for scale_index in range(len(scale_windows)):
        embeddings = scale_embeddings[scale_index]
        embedding_parts.append(embeddings)
        segment_counts.append(embeddings.shape[0])
        if scale_index == base_scale_index:
            timestamps = torch_mod.tensor([[row.start, row.end] for row in rows], dtype=torch_mod.float32)
        else:
            timestamps = torch_mod.tensor(
                [[window.start, window.end] for window in windows_by_scale[scale_index]],
                dtype=torch_mod.float32,
            )
        timestamp_parts.append(timestamps)

    return {
        file_id: {
            "multiscale_weights": torch_mod.ones(1, len(scale_windows), dtype=torch_mod.float32),
            "embeddings": torch_mod.cat(embedding_parts, dim=0),
            "timestamps": torch_mod.cat(timestamp_parts, dim=0),
            "multiscale_segment_counts": torch_mod.tensor(segment_counts, dtype=torch_mod.long),
        }
    }


def run_nemo_clustering(
    file_id: str,
    audio_path: Path,
    rows: list[SubtitleRow],
    windows_by_scale: dict[int, list[RowScaleWindow]],
    scale_embeddings: dict[int, Any],
    scale_windows: list[float],
    output_dir: Path,
    max_num_speakers: int,
    num_speakers: int | None,
    max_rp_threshold: float,
    sparse_search_volume: int,
    enhanced_count_thres: int,
    chunk_cluster_count: int,
    embeddings_per_chunk: int,
    device: str,
    progress: ProgressLogger | None = None,
) -> list[int]:
    torch_mod = require_torch()
    try:
        from nemo.collections.asr.parts.utils.speaker_utils import perform_clustering
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR and OmegaConf are required for NeMo clustering. Install "
            "`nemo_toolkit[asr]` before running the full pipeline."
        ) from exc

    raw_dir = output_dir / "nemo_raw"
    pred_rttm_dir = raw_dir / "pred_rttms"
    pred_rttm_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "speaker_outputs").mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.log(
            f"building NeMo clustering inputs for {len(rows)} rows across "
            f"{len(scale_windows)} scales"
        )
    embs_and_timestamps = build_clustering_inputs(
        file_id=file_id,
        rows=rows,
        windows_by_scale=windows_by_scale,
        scale_embeddings=scale_embeddings,
        scale_windows=scale_windows,
    )
    audio_rttm_map = {
        file_id: {
            "audio_filepath": str(audio_path.resolve()),
            "rttm_filepath": None,
            "offset": 0,
            "duration": None,
            "text": "-",
            "num_speakers": num_speakers,
            "uem_filepath": None,
            "ctm_filepath": None,
        }
    }
    clustering_params = OmegaConf.create(
        {
            "oracle_num_speakers": num_speakers is not None,
            "max_num_speakers": max_num_speakers,
            "enhanced_count_thres": enhanced_count_thres,
            "enhanced_count_threshold": enhanced_count_thres,
            "max_rp_threshold": max_rp_threshold,
            "sparse_search_volume": sparse_search_volume,
            "chunk_cluster_count": chunk_cluster_count,
            "embeddings_per_chunk": embeddings_per_chunk,
        }
    )
    if progress is not None:
        speaker_mode = (
            f"oracle num_speakers={num_speakers}"
            if num_speakers is not None
            else f"auto speaker count, max={max_num_speakers}"
        )
        progress.log(f"running NeMo spectral clustering ({speaker_mode})")
    perform_clustering(
        embs_and_timestamps=embs_and_timestamps,
        AUDIO_RTTM_MAP=audio_rttm_map,
        out_rttm_dir=str(pred_rttm_dir),
        clustering_params=clustering_params,
        device=torch_mod.device(device),
        verbose=True,
    )

    label_path = raw_dir / "speaker_outputs" / f"subsegments_scale{len(scale_windows) - 1}_cluster.label"
    if progress is not None:
        progress.log(f"reading NeMo cluster labels: {label_path}")
    labels = parse_cluster_label_file(label_path, file_id, len(rows))
    if progress is not None:
        progress.log(f"NeMo clustering complete: {len(set(labels))} clusters for {len(labels)} rows")
    return labels


def parse_float_token(value: str) -> float:
    value = value.strip()
    if value.startswith("tensor(") and value.endswith(")"):
        value = value[len("tensor(") : -1]
    return float(value)


def parse_cluster_id(value: str, fallback_map: dict[str, int]) -> int:
    value = value.strip()
    suffix = value.rsplit("_", 1)[-1]
    if suffix.lstrip("-").isdigit():
        return int(suffix)
    if value not in fallback_map:
        fallback_map[value] = len(fallback_map)
    return fallback_map[value]


def parse_cluster_label_file(path: Path, file_id: str, expected_count: int) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"NeMo cluster label file not found: {path}")
    labels: list[int] = []
    fallback_map: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == file_id and len(parts) >= 4:
            _start = parse_float_token(parts[1])
            _end = parse_float_token(parts[2])
            labels.append(parse_cluster_id(parts[3], fallback_map))
        elif len(parts) >= 3:
            _start = parse_float_token(parts[0])
            _end = parse_float_token(parts[1])
            labels.append(parse_cluster_id(parts[2], fallback_map))
    if len(labels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} base-scale cluster labels in {path}, got {len(labels)}"
        )
    return labels


def cluster_centroids_by_scale(scale_embeddings: dict[int, Any], labels: list[int]) -> dict[int, dict[int, Any]]:
    torch_mod = require_torch()
    label_tensor = torch_mod.tensor(labels, dtype=torch_mod.long)
    centroids: dict[int, dict[int, Any]] = {}
    for scale_index, embeddings in scale_embeddings.items():
        by_cluster = {}
        for cluster_id in sorted(set(labels)):
            mask = label_tensor == cluster_id
            centroid = embeddings[mask].mean(dim=0)
            by_cluster[cluster_id] = l2_normalize_torch(centroid)
        centroids[scale_index] = by_cluster
    return centroids


def row_similarity_details(
    base_embeddings: Any,
    labels: list[int],
    centroids: dict[int, Any],
) -> tuple[list[dict[str, float | int | None]], list[int | None]]:
    torch_mod = require_torch()
    details = []
    competitors = []
    for index, embedding in enumerate(base_embeddings):
        own_cluster = labels[index]
        scored = [(cluster_id, float(torch_mod.dot(embedding, centroid))) for cluster_id, centroid in centroids.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        own_similarity = next(score for cluster_id, score in scored if cluster_id == own_cluster)
        competitor = next((cluster_id for cluster_id, _score in scored if cluster_id != own_cluster), None)
        second_similarity = next((score for cluster_id, score in scored if cluster_id != own_cluster), None)
        competitors.append(competitor)
        details.append(
            {
                "best_similarity": own_similarity,
                "second_best_similarity": second_similarity,
                "margin": own_similarity - second_similarity if second_similarity is not None else 1.0,
                "nearest_competitor_cluster": competitor,
            }
        )
    return details, competitors


def iter_blocks(row_count: int, block_size: int, overlap: int) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if overlap < 0 or overlap >= block_size:
        raise ValueError("overlap must be >= 0 and smaller than block_size")
    step = block_size - overlap
    blocks = []
    start = 0
    while start < row_count:
        end = min(row_count, start + block_size)
        blocks.append((start, end))
        if end == row_count:
            break
        start += step
    return blocks


def candidate_pairs_for_block(
    labels: list[int],
    competitors: list[int | None],
    start: int,
    end: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for index in range(start, end):
        competitor = competitors[index]
        if competitor is None or competitor == labels[index]:
            continue
        counts[tuple(sorted((labels[index], competitor)))] += 1
    return [pair for pair, _count in counts.most_common(max_pairs)]


def load_msdd_model(model_path: str, device: str):
    torch_mod = require_torch()
    try:
        from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
    except ImportError as exc:
        raise RuntimeError(
            "NeMo MSDD refinement requires NeMo ASR. Install `nemo_toolkit[asr]` "
            "or run with `--neural-refinement off`."
        ) from exc

    map_location = torch_mod.device(device)
    if model_path.endswith(".nemo"):
        model = EncDecDiarLabelModel.restore_from(model_path, map_location=map_location)
    elif model_path.endswith(".ckpt"):
        model = EncDecDiarLabelModel.load_from_checkpoint(model_path, map_location=map_location)
    else:
        model = EncDecDiarLabelModel.from_pretrained(
            model_name=model_path,
            map_location=map_location,
        )
    model = model.to(map_location)
    model.eval()
    return model


def run_msdd_pair(msdd_model: Any, block_embeddings: Any, pair_centroids: Any, device: str):
    torch_mod = require_torch()
    np_mod = require_numpy()
    input_signal = block_embeddings.unsqueeze(0).float().to(device)
    input_length = torch_mod.tensor([block_embeddings.shape[0]], dtype=torch_mod.long, device=device)
    emb_vectors = pair_centroids.unsqueeze(0).float().to(device)
    targets = torch_mod.zeros(
        (1, block_embeddings.shape[0], pair_centroids.shape[-1]),
        dtype=torch_mod.float32,
        device=device,
    )
    with torch_mod.no_grad():
        if hasattr(msdd_model, "forward_infer"):
            preds, _scale_weights = msdd_model.forward_infer(
                input_signal=input_signal,
                input_signal_length=input_length,
                emb_vectors=emb_vectors,
                targets=targets,
            )
        else:
            preds, _scale_weights = msdd_model.msdd(
                ms_emb_seq=input_signal,
                length=input_length,
                ms_avg_embs=emb_vectors,
                targets=targets,
            )
    preds = preds.detach().float().cpu()
    if float(preds.min()) < 0.0 or float(preds.max()) > 1.0:
        preds = torch_mod.sigmoid(preds)
    return np_mod.asarray(preds.squeeze(0).numpy())


def run_local_pairwise_refinement(
    rows: list[SubtitleRow],
    labels: list[int],
    scale_embeddings: dict[int, Any],
    msdd_model_path: str,
    device: str,
    block_size: int,
    block_overlap: int,
    max_pairs_per_block: int,
    relabel_threshold: float,
    relabel_margin: float,
    progress_every: int,
    progress: ProgressLogger | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    torch_mod = require_torch()
    if progress is not None:
        progress.log("preparing MSDD local-pairwise refinement candidates")
    centroids = cluster_centroids_by_scale(scale_embeddings, labels)
    base_scale_index = max(scale_embeddings)
    similarity_details, competitors = row_similarity_details(
        scale_embeddings[base_scale_index],
        labels,
        centroids[base_scale_index],
    )
    blocks = iter_blocks(len(rows), block_size, block_overlap)
    block_pairs = [
        (
            block_start,
            block_end,
            candidate_pairs_for_block(labels, competitors, block_start, block_end, max_pairs_per_block),
        )
        for block_start, block_end in blocks
    ]
    block_pairs = [item for item in block_pairs if item[2]]
    total_pairs = sum(len(pairs) for _start, _end, pairs in block_pairs)
    if progress is not None:
        progress.log(
            f"MSDD candidate scan complete: {len(block_pairs)}/{len(blocks)} blocks "
            f"contain {total_pairs} candidate pairs"
        )
    if not block_pairs:
        audit_rows = [
            {
                "row_index": row.row_index,
                "original_cluster": labels[index],
                "nearest_competitor_cluster": competitors[index],
                "final_cluster": labels[index],
                "action": "no_pairwise_candidate",
                **similarity_details[index],
            }
            for index, row in enumerate(rows)
        ]
        stats = {
            "enabled": True,
            "model": msdd_model_path,
            "mode": "local-pairwise",
            "blocks": len(blocks),
            "pairs_run": 0,
            "relabeled_rows": 0,
            "relabel_threshold": relabel_threshold,
            "relabel_margin": relabel_margin,
        }
        if progress is not None:
            progress.log("MSDD refinement skipped: no pairwise candidates were found")
        return list(labels), audit_rows, stats

    if progress is not None:
        progress.log(f"loading MSDD model '{msdd_model_path}' on {device}")
    msdd_model = load_msdd_model(msdd_model_path, device)
    if progress is not None:
        progress.log(f"MSDD model ready: {msdd_model_path}")

    probability_sums: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    probability_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pairs_run = 0

    for block_number, (block_start, block_end, pairs) in enumerate(block_pairs, start=1):
        if progress is not None:
            progress.log(
                f"MSDD block {block_number}/{len(block_pairs)}: rows "
                f"{block_start + 1}-{block_end}, {len(pairs)} candidate pairs"
            )
        block_embeddings = torch_mod.stack(
            [scale_embeddings[scale_index][block_start:block_end] for scale_index in sorted(scale_embeddings)],
            dim=1,
        )
        for left_cluster, right_cluster in pairs:
            pair_centroids = torch_mod.stack(
                [
                    torch_mod.stack(
                        [centroids[scale_index][left_cluster], centroids[scale_index][right_cluster]],
                        dim=-1,
                    )
                    for scale_index in sorted(scale_embeddings)
                ],
                dim=0,
            )
            probabilities = run_msdd_pair(msdd_model, block_embeddings, pair_centroids, device)
            pairs_run += 1
            if (
                progress is not None
                and progress_every > 0
                and (pairs_run == 1 or pairs_run % progress_every == 0 or pairs_run == total_pairs)
            ):
                progress.log(
                    f"MSDD pair inference {pairs_run}/{total_pairs}: "
                    f"clusters {left_cluster} vs {right_cluster}"
                )
            for offset, row_index in enumerate(range(block_start, block_end)):
                probability_sums[row_index][left_cluster] += float(probabilities[offset, 0])
                probability_counts[row_index][left_cluster] += 1
                probability_sums[row_index][right_cluster] += float(probabilities[offset, 1])
                probability_counts[row_index][right_cluster] += 1

    refined = list(labels)
    audit_rows = []
    relabeled = 0
    for index, row in enumerate(rows):
        own_cluster = labels[index]
        competitor = competitors[index]

        def average_probability(cluster_id: int | None) -> float | None:
            if cluster_id is None or probability_counts[index].get(cluster_id, 0) == 0:
                return None
            return probability_sums[index][cluster_id] / probability_counts[index][cluster_id]

        own_probability = average_probability(own_cluster)
        competitor_probability = average_probability(competitor)
        action = "keep"
        if (
            competitor is not None
            and own_probability is not None
            and competitor_probability is not None
            and competitor_probability >= relabel_threshold
            and competitor_probability - own_probability >= relabel_margin
        ):
            refined[index] = competitor
            relabeled += 1
            action = "relabel"
        elif competitor is None:
            action = "no_competitor"
        elif own_probability is None or competitor_probability is None:
            action = "no_msdd_probability"

        audit_rows.append(
            {
                "row_index": row.row_index,
                "original_cluster": own_cluster,
                "nearest_competitor_cluster": competitor,
                "original_probability": round(own_probability, 6) if own_probability is not None else None,
                "competitor_probability": (
                    round(competitor_probability, 6) if competitor_probability is not None else None
                ),
                "final_cluster": refined[index],
                "action": action,
                **similarity_details[index],
            }
        )

    stats = {
        "enabled": True,
        "model": msdd_model_path,
        "mode": "local-pairwise",
        "blocks": len(blocks),
        "pairs_run": pairs_run,
        "relabeled_rows": relabeled,
        "relabel_threshold": relabel_threshold,
        "relabel_margin": relabel_margin,
    }
    if progress is not None:
        progress.log(f"MSDD refinement complete: {pairs_run} pairs run, {relabeled} rows relabeled")
    return refined, audit_rows, stats


def disabled_refinement_audit(
    rows: list[SubtitleRow],
    labels: list[int],
    scale_embeddings: dict[int, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, float | int | None]]]:
    centroids = cluster_centroids_by_scale(scale_embeddings, labels)
    base_scale_index = max(scale_embeddings)
    similarity_details, competitors = row_similarity_details(
        scale_embeddings[base_scale_index],
        labels,
        centroids[base_scale_index],
    )
    audit_rows = [
        {
            "row_index": row.row_index,
            "original_cluster": labels[index],
            "nearest_competitor_cluster": competitors[index],
            "final_cluster": labels[index],
            "action": "disabled",
            **similarity_details[index],
        }
        for index, row in enumerate(rows)
    ]
    stats = {
        "enabled": False,
        "mode": "off",
        "blocks": 0,
        "pairs_run": 0,
        "relabeled_rows": 0,
    }
    return audit_rows, stats, similarity_details


def color_key(color: str | None) -> str | None:
    if color is None:
        return None
    normalized = str(color).strip().lower()
    if normalized in NON_ACTIONABLE_COLOR_KEYS:
        return None
    return normalized


def dominant_label_for_run(rows: list[SubtitleRow], labels: list[int], start_index: int, end_index: int) -> int:
    durations: defaultdict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    for index in range(start_index, end_index):
        label = labels[index]
        durations[label] += rows[index].duration
        first_seen.setdefault(label, index)
    return min(durations, key=lambda label: (-durations[label], first_seen[label], label))


def apply_adjacent_color_constraints(
    rows: list[SubtitleRow],
    labels: list[int],
    progress: ProgressLogger | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    refined = list(labels)
    audit_rows: list[dict[str, Any]] = []
    changed_row_indices: list[int] = []
    constrained_runs = 0
    skipped_non_actionable_rows = 0
    index = 0

    while index < len(rows):
        key = color_key(rows[index].color)
        run_start = index
        index += 1
        while index < len(rows) and color_key(rows[index].color) == key:
            index += 1
        run_end = index

        if key is None:
            skipped_non_actionable_rows += run_end - run_start
            continue
        if run_end - run_start < 2:
            continue

        original_run_labels = refined[run_start:run_end]
        target_label = dominant_label_for_run(rows, refined, run_start, run_end)
        changed_rows = []
        for row_index in range(run_start, run_end):
            if refined[row_index] != target_label:
                refined[row_index] = target_label
                changed_rows.append(rows[row_index].row_index)

        constrained_runs += 1
        changed_row_indices.extend(changed_rows)
        audit_rows.append(
            {
                "action": "merge_adjacent_same_color",
                "color": rows[run_start].color,
                "normalized_color": key,
                "start_row_index": rows[run_start].row_index,
                "end_row_index": rows[run_end - 1].row_index,
                "row_indices": [row.row_index for row in rows[run_start:run_end]],
                "original_clusters": original_run_labels,
                "final_cluster": target_label,
                "changed_row_indices": changed_rows,
                "cluster_counts": dict(Counter(original_run_labels)),
                "cluster_durations": {
                    str(label): round(
                        sum(
                            rows[row_index].duration
                            for row_index in range(run_start, run_end)
                            if original_run_labels[row_index - run_start] == label
                        ),
                        6,
                    )
                    for label in sorted(set(original_run_labels))
                },
            }
        )

    stats = {
        "enabled": True,
        "mode": "adjacent-same-color-run",
        "constrained_runs": constrained_runs,
        "changed_rows": len(changed_row_indices),
        "changed_row_indices": changed_row_indices,
        "skipped_non_actionable_rows": skipped_non_actionable_rows,
        "ignored_color_keys": sorted(NON_ACTIONABLE_COLOR_KEYS),
    }
    if progress is not None:
        progress.log(
            f"color run constraint complete: {constrained_runs} same-color runs, "
            f"{len(changed_row_indices)} rows changed"
        )
    return refined, audit_rows, stats


def stable_speaker_ids(labels: list[int], rows: list[SubtitleRow]) -> dict[int, int]:
    indexed = []
    for cluster_id in sorted(set(labels)):
        member_rows = [row for row, label in zip(rows, labels) if label == cluster_id]
        first_start = min(row.start for row in member_rows)
        total_duration = sum(row.duration for row in member_rows)
        indexed.append((first_start, -total_duration, cluster_id))
    indexed.sort()
    return {cluster_id: speaker_id for speaker_id, (_first, _duration, cluster_id) in enumerate(indexed, 1)}


def rttm_line(file_id: str, start: float, duration: float, speaker_label: str) -> str:
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>"


def confidence_from_similarity(detail: dict[str, Any]) -> float:
    best = detail.get("best_similarity")
    if best is None:
        return 0.0
    return round(float(max(0.0, min(1.0, (float(best) + 1.0) / 2.0))), 4)


def write_outputs(
    output_dir: Path,
    output_rttm: Path,
    rows: list[SubtitleRow],
    labels: list[int],
    speaker_ids: dict[int, int],
    similarity_details: list[dict[str, Any]],
    refinement_stats: dict[str, Any],
    color_refinement_stats: dict[str, Any],
    scale_windows: list[float],
    speaker_model: str,
    msdd_model: str,
    rttm_file_id: str,
    clustering_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    line_rows = []
    rttm_rows = []
    color_changed_rows = set(color_refinement_stats.get("changed_row_indices", []))
    for row, cluster_id, detail in zip(rows, labels, similarity_details):
        speaker_id = speaker_ids[cluster_id]
        speaker_label = f"speaker_{speaker_id:03d}"
        if row.row_index in color_changed_rows:
            assignment_kind = "nemo_color_constrained"
        elif refinement_stats.get("enabled") and refinement_stats.get("relabeled_rows", 0) > 0:
            assignment_kind = "nemo_msdd_refined"
        else:
            assignment_kind = "nemo_clustered"
        output_row = {
            "unit_id": row.row_index,
            "row_index": row.row_index,
            "cue_index": row.row_index,
            "span_index": 1,
            "source_type": "json",
            "segment_id": row.row_index,
            "tracklet_id": row.row_index,
            "speaker_id": speaker_id,
            "speaker_label": speaker_label,
            "cluster_id": cluster_id,
            "confidence": confidence_from_similarity(detail),
            "assignment_kind": assignment_kind,
            "color_constraint_applied": row.row_index in color_changed_rows,
            "best_similarity": detail.get("best_similarity"),
            "second_best_similarity": detail.get("second_best_similarity"),
            "margin": detail.get("margin"),
            "start": row.start,
            "end": row.end,
            "duration": row.duration,
            "color": row.color,
            "text": row.text,
            "gold_speaker": row.gold_speaker,
            "color_cue_confidence": row.color_cue_confidence,
            "color_cue_ambiguous": row.color_cue_ambiguous,
            "source_units": [row.source_unit()],
        }
        line_rows.append(output_row)
        rttm_rows.append(rttm_line(rttm_file_id, row.start, row.duration, speaker_label))

    cluster_rows = []
    for cluster_id, speaker_id in sorted(speaker_ids.items(), key=lambda item: item[1]):
        member_indices = [index for index, label in enumerate(labels) if label == cluster_id]
        member_rows = [rows[index] for index in member_indices]
        cluster_rows.append(
            {
                "speaker_id": speaker_id,
                "speaker_label": f"speaker_{speaker_id:03d}",
                "cluster_id": cluster_id,
                "tracklet_count": len(member_rows),
                "segment_count": len(member_rows),
                "total_duration": round(sum(row.duration for row in member_rows), 6),
                "first_start": min(row.start for row in member_rows),
                "last_end": max(row.end for row in member_rows),
                "colors": sorted({row.color for row in member_rows}),
                "row_indices": [row.row_index for row in member_rows],
            }
        )

    summary = {
        "segments": len(rows),
        "lines": len(rows),
        "tracklets": len(rows),
        "clusters": len(cluster_rows),
        "unknown_segments": 0,
        "unknown_lines": 0,
        "rttm_turns": len(rttm_rows),
        "source_kind": "json",
        "pipeline": "nemo_exact_row",
        "speaker_model": speaker_model,
        "msdd_model": msdd_model,
        "scale_windows": scale_windows,
        "base_scale_index": len(scale_windows) - 1,
        "clustering": clustering_config,
        "neural_refinement": refinement_stats,
        "color_refinement": color_refinement_stats,
    }

    write_jsonl(output_dir / "speaker_lines.jsonl", line_rows)
    write_jsonl(output_dir / "speaker_segments.jsonl", line_rows)
    (output_dir / "speaker_clusters.json").write_text(
        json.dumps(cluster_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rttm_text = "\n".join(rttm_rows) + "\n"
    (output_dir / "speaker_segments.rttm").write_text(rttm_text, encoding="utf-8")
    (output_dir / "speaker_json_rows.rttm").write_text(rttm_text, encoding="utf-8")
    output_rttm.parent.mkdir(parents=True, exist_ok=True)
    output_rttm.write_text(rttm_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    progress = ProgressLogger()
    progress.log("starting NeMo exact-row diarization pipeline")
    progress.log(f"input subtitle: {args.subtitle}")
    progress.log(f"input video: {args.video}")
    rows = parse_subtitle_rows(args.subtitle, max_rows=args.max_rows)
    if not rows:
        raise ValueError(f"No subtitle rows found in {args.subtitle}")
    scale_windows = parse_scale_windows(args.scale_windows)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress.log(f"loaded {len(rows)} subtitle rows spanning {rows[0].start:.3f}s to {rows[-1].end:.3f}s")
    progress.log(f"scale windows: {scale_windows}")
    progress.log(f"embedding dir: {args.embedding_dir}")
    progress.log(f"output dir: {args.output_dir}")
    progress.log(f"output RTTM: {args.output_rttm}")

    if args.dry_run:
        progress.log("dry run: building row-centered windows without NeMo inference")
        audio_duration = max(row.end for row in rows)
        windows_by_scale = build_scale_windows(rows, scale_windows, audio_duration=audio_duration, min_window=args.min_window)
        row_segment_rows = row_segments_metadata(rows, windows_by_scale)
        write_jsonl(args.embedding_dir / "row_segments.jsonl", row_segment_rows)
        write_jsonl(args.output_dir / "row_segments.jsonl", row_segment_rows)
        progress.log("dry run row metadata written")
        print(f"rows: {len(rows)}")
        print(f"scale_windows: {scale_windows}")
        print(f"dry_run_audio_duration: {audio_duration:.3f}")
        print(f"wrote: {args.embedding_dir / 'row_segments.jsonl'}")
        print("Dry run complete; no audio, NeMo embeddings, clustering, or MSDD were run.")
        return

    device = choose_device(args.device)
    progress.log(f"using device: {device}")
    audio_cache = args.audio_cache or args.embedding_dir / f"audio_{args.sample_rate}hz_mono.wav"
    extract_audio(args.video, audio_cache, args.sample_rate, args.overwrite_audio, progress=progress)
    audio_duration = wav_duration(audio_cache)
    progress.log(f"audio duration: {audio_duration:.3f}s")
    windows_by_scale = build_scale_windows(rows, scale_windows, audio_duration=audio_duration, min_window=args.min_window)
    progress.log(f"built {len(scale_windows)} embedding scales with {len(rows) * len(scale_windows)} total row windows")
    row_segment_rows = row_segments_metadata(rows, windows_by_scale)
    write_jsonl(args.embedding_dir / "row_segments.jsonl", row_segment_rows)
    write_jsonl(args.output_dir / "row_segments.jsonl", row_segment_rows)
    progress.log("wrote row segment metadata")

    scale_embeddings = extract_multiscale_embeddings(
        rows=rows,
        windows_by_scale=windows_by_scale,
        audio_path=audio_cache,
        embedding_dir=args.embedding_dir,
        speaker_model=args.speaker_model,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        file_id=args.rttm_file_id,
        progress_every=args.progress_every,
        progress=progress,
    )
    save_embedding_package(
        args.embedding_dir / "row_multiscale_embeddings.pt",
        rows,
        windows_by_scale,
        scale_embeddings,
        scale_windows,
        args.speaker_model,
        audio_cache,
    )
    progress.log(f"saved embedding package: {args.embedding_dir / 'row_multiscale_embeddings.pt'}")

    cluster_labels = run_nemo_clustering(
        file_id=args.rttm_file_id,
        audio_path=audio_cache,
        rows=rows,
        windows_by_scale=windows_by_scale,
        scale_embeddings=scale_embeddings,
        scale_windows=scale_windows,
        output_dir=args.output_dir,
        max_num_speakers=args.max_num_speakers,
        num_speakers=args.num_speakers,
        max_rp_threshold=args.max_rp_threshold,
        sparse_search_volume=args.sparse_search_volume,
        enhanced_count_thres=args.enhanced_count_thres,
        chunk_cluster_count=args.chunk_cluster_count,
        embeddings_per_chunk=args.embeddings_per_chunk,
        device=device,
        progress=progress,
    )

    if args.neural_refinement == "local-pairwise":
        final_labels, refinement_rows, refinement_stats = run_local_pairwise_refinement(
            rows=rows,
            labels=cluster_labels,
            scale_embeddings=scale_embeddings,
            msdd_model_path=args.msdd_model,
            device=device,
            block_size=args.msdd_block_size,
            block_overlap=args.msdd_block_overlap,
            max_pairs_per_block=args.msdd_max_pairs_per_block,
            relabel_threshold=args.msdd_relabel_threshold,
            relabel_margin=args.msdd_relabel_margin,
            progress_every=args.progress_every,
            progress=progress,
        )
    else:
        progress.log("MSDD neural refinement disabled")
        final_labels = cluster_labels
        refinement_rows, refinement_stats, _unused_details = disabled_refinement_audit(rows, final_labels, scale_embeddings)

    if args.disable_color_run_merge:
        progress.log("adjacent same-color run constraint disabled")
        color_refinement_rows: list[dict[str, Any]] = []
        color_refinement_stats = {
            "enabled": False,
            "mode": "off",
            "constrained_runs": 0,
            "changed_rows": 0,
            "changed_row_indices": [],
        }
    else:
        progress.log("applying adjacent same-color run constraint")
        final_labels, color_refinement_rows, color_refinement_stats = apply_adjacent_color_constraints(
            rows,
            final_labels,
            progress=progress,
        )

    final_similarity_details, _competitors = row_similarity_details(
        scale_embeddings[max(scale_embeddings)],
        final_labels,
        cluster_centroids_by_scale(scale_embeddings, final_labels)[max(scale_embeddings)],
    )

    progress.log(f"writing MSDD audit rows: {args.output_dir / 'msdd_refinement.jsonl'}")
    write_jsonl(args.output_dir / "msdd_refinement.jsonl", refinement_rows)
    progress.log(f"writing color constraint audit rows: {args.output_dir / 'color_refinement.jsonl'}")
    write_jsonl(args.output_dir / "color_refinement.jsonl", color_refinement_rows)
    speaker_ids = stable_speaker_ids(final_labels, rows)
    progress.log(f"assigned stable speaker labels for {len(speaker_ids)} clusters")
    clustering_config = {
        "oracle_num_speakers": args.num_speakers is not None,
        "num_speakers": args.num_speakers,
        "max_num_speakers": args.max_num_speakers,
        "max_rp_threshold": args.max_rp_threshold,
        "sparse_search_volume": args.sparse_search_volume,
        "enhanced_count_thres": args.enhanced_count_thres,
        "chunk_cluster_count": args.chunk_cluster_count,
        "embeddings_per_chunk": args.embeddings_per_chunk,
    }
    write_outputs(
        output_dir=args.output_dir,
        output_rttm=args.output_rttm,
        rows=rows,
        labels=final_labels,
        speaker_ids=speaker_ids,
        similarity_details=final_similarity_details,
        refinement_stats=refinement_stats,
        color_refinement_stats=color_refinement_stats,
        scale_windows=scale_windows,
        speaker_model=args.speaker_model,
        msdd_model=args.msdd_model,
        rttm_file_id=args.rttm_file_id,
        clustering_config=clustering_config,
    )
    progress.log(f"final speaker outputs written, RTTM: {args.output_rttm}")

    print(f"rows: {len(rows)}")
    print(f"clusters: {len(speaker_ids)}")
    print(f"neural_refinement: {args.neural_refinement}")
    print(f"msdd_relabeled_rows: {refinement_stats.get('relabeled_rows', 0)}")
    print(f"color_constrained_runs: {color_refinement_stats.get('constrained_runs', 0)}")
    print(f"color_changed_rows: {color_refinement_stats.get('changed_rows', 0)}")
    print(f"wrote: {args.embedding_dir / 'row_multiscale_embeddings.pt'}")
    print(f"wrote: {args.output_dir}")
    print(f"wrote_rttm: {args.output_rttm}")


if __name__ == "__main__":
    main()
