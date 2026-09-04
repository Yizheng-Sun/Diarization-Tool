#!/usr/bin/env python3
"""Tracklet-level subtitle diarization with NVIDIA TitaNet embeddings.

This pipeline differs from ``nemo_subtitle_diarize.py`` in one important way:
it clusters adjacent same-color subtitle rows as tracklets rather than
clustering individual subtitle rows. Tracklet embeddings are weighted averages
of their row embeddings, and adjacent tracklets at color-change speaker turns
are used as cannot-link constraints during clustering.
"""

from __future__ import annotations

import argparse
import html
import json
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
    import torch
except ImportError:  # pragma: no cover - dry-run/tests do not require torch.
    torch = None


DEFAULT_JSON = Path("data/labels/American_Fiction_2023_with_colors.json")
DEFAULT_VIDEO = Path("data/movies/American_Fiction_2023.mp4")
DEFAULT_OUTPUT_RTTM = Path("data/labels/American_Fiction_2023_titanet_tracklet_pred.rttm")
DEFAULT_OUTPUT_DIR = Path("data/clusters/American_Fiction_2023_titanet_tracklet")
DEFAULT_EMBEDDING_DIR = Path("data/embeddings/American_Fiction_2023_titanet_tracklet")
DEFAULT_RTTM_FILE_ID = "American_Fiction_2023"
DEFAULT_SPEAKER_MODEL = "nvidia/speakerverification_en_titanet_large"

TAG_RE = re.compile(r"<[^>]+>")
VTT_TIMING_RE = re.compile(r"^\s*(\d\d:\d\d:\d\d(?:\.\d+)?)\s+-->\s+(\d\d:\d\d:\d\d(?:\.\d+)?)")
VTT_SPAN_RE = re.compile(r"<c(?:\.([A-Za-z0-9_-]+))?[^>]*>(.*?)</c>", re.DOTALL)
NON_ACTIONABLE_COLOR_KEYS = {"", "unknown", "none", "null", "nan", "n/a", "na"}


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch and NeMo are required for TitaNet embedding extraction. "
            "Install the project environment before running model inference."
        )
    return torch


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started_at = time.monotonic()

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[progress {format_elapsed(time.monotonic() - self.started_at)}] {message}", flush=True)


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


@dataclass(frozen=True)
class EmbeddingWindow:
    row_index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Tracklet:
    tracklet_id: int
    row_indices: list[int]
    row_positions: list[int]
    color: str
    normalized_color: str | None
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class LocalColorWindow:
    window_id: int
    tracklet_indices: list[int]
    colors: list[str]
    start: float
    end: float
    span: float
    max_gap: float


@dataclass(frozen=True)
class ColorRoleUnit:
    unit_id: int
    tracklet_indices: list[int]
    normalized_color: str | None
    start: float
    end: float
    duration: float
    row_count: int
    window_ids: list[int]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


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
            "Extract segment embeddings with nvidia/speakerverification_en_titanet_large, "
            "merge adjacent same-color segments into tracklets, and cluster tracklets "
            "with adjacent color-change cannot-link constraints."
        )
    )
    parser.add_argument("--subtitle", "--json", dest="subtitle", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-rttm", type=Path, default=DEFAULT_OUTPUT_RTTM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    parser.add_argument("--num-speakers", type=parse_optional_int, default=None)
    parser.add_argument("--min-num-speakers", type=int, default=2)
    parser.add_argument("--max-num-speakers", type=int, default=80)
    parser.add_argument(
        "--clustering-method",
        choices=("constrained-spectral", "constrained-agglomerative"),
        default="constrained-spectral",
        help="Tracklet clustering backend.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.35,
        help="Cosine-distance threshold used by constrained agglomerative clustering.",
    )
    parser.add_argument(
        "--spectral-neighbors",
        type=int,
        default=30,
        help="Number of nearest-neighbor affinity edges to keep per tracklet. Use 0 for dense.",
    )
    parser.add_argument(
        "--spectral-sigma",
        type=float,
        default=0.15,
        help="RBF sigma applied to cosine distances when building spectral affinities.",
    )
    parser.add_argument(
        "--spectral-kmeans-iters",
        type=int,
        default=100,
        help="Maximum k-means iterations in spectral embedding space.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--audio-cache", type=Path)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--rttm-file-id", default=DEFAULT_RTTM_FILE_ID)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--min-embedding-duration",
        type=float,
        default=0.5,
        help="Pad short subtitle rows to at least this duration for embedding extraction.",
    )
    parser.add_argument(
        "--short-tracklet-mode",
        choices=("assign-after", "include"),
        default="assign-after",
        help=(
            "In assign-after mode, short single-segment tracklets are held out of clustering "
            "and assigned to fixed anchor centroids afterward."
        ),
    )
    parser.add_argument(
        "--anchor-min-duration",
        type=float,
        default=1.5,
        help="Minimum tracklet duration for anchor clustering in assign-after mode.",
    )
    parser.add_argument(
        "--anchor-min-segments",
        type=int,
        default=2,
        help="Minimum subtitle-row count for anchor clustering in assign-after mode.",
    )
    parser.add_argument(
        "--weak-assignment-threshold",
        type=float,
        default=0.45,
        help=(
            "Maximum cosine distance for assigning a weak tracklet to an anchor centroid. "
            "Weak tracklets farther than this become singleton clusters; use a negative "
            "value to always assign to the nearest allowed centroid."
        ),
    )
    parser.add_argument(
        "--disable-local-color-windows",
        action="store_true",
        help="Disable local color-role windows and cluster raw tracklets directly.",
    )
    parser.add_argument(
        "--local-color-window-max-gap",
        type=float,
        default=2.0,
        help="Maximum gap in seconds between adjacent tracklets inside a local color-role window.",
    )
    parser.add_argument(
        "--local-color-window-max-span",
        type=float,
        default=30.0,
        help="Maximum time span in seconds for one local color-role window.",
    )
    parser.add_argument(
        "--local-color-window-max-colors",
        type=int,
        default=3,
        help="Maximum number of usable subtitle colors inside one local color-role window.",
    )
    parser.add_argument(
        "--local-color-window-min-turns",
        type=int,
        default=3,
        help="Minimum number of tracklets required before a local color-role window is trusted.",
    )
    parser.add_argument(
        "--merge-unknown-colors",
        action="store_true",
        help="Allow adjacent unknown-color rows to form one tracklet.",
    )
    parser.add_argument(
        "--disable-cannot-link",
        action="store_true",
        help="Disable adjacent color-change cannot-link constraints.",
    )
    return parser.parse_args()


def seconds_from_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def plain_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def seconds_from_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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
        color, confidence, ambiguous = parse_vtt_cue_color(cue_text)
        parsed.append(
            SubtitleRow(
                row_index=len(parsed) + 1,
                start=start,
                end=end,
                color=color,
                text=plain_text(cue_text),
                color_cue_confidence=confidence,
                color_cue_ambiguous=ambiguous,
            )
        )
        if max_rows is not None and len(parsed) >= max_rows:
            break
    return parsed


def color_key(color: str | None) -> str | None:
    if color is None:
        return None
    normalized = str(color).strip().lower()
    if normalized in NON_ACTIONABLE_COLOR_KEYS:
        return None
    return normalized


def parse_subtitle_rows(subtitle_path: Path, max_rows: int | None = None) -> list[SubtitleRow]:
    if subtitle_path.suffix.lower() in {".vtt", ".webvtt"}:
        return parse_vtt_subtitle_rows(subtitle_path, max_rows=max_rows)

    payload = json.loads(subtitle_path.read_text(encoding="utf-8"))
    rows = payload.get("mappings_compact")
    if not isinstance(rows, list):
        raise ValueError(f"Expected {subtitle_path} to contain list key 'mappings_compact'")

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
        if progress:
            progress.log(f"audio cache found: {audio_path}")
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
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
    if progress:
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


def build_embedding_windows(
    rows: list[SubtitleRow],
    audio_duration: float,
    min_embedding_duration: float,
) -> list[EmbeddingWindow]:
    windows = []
    for row in rows:
        duration = max(row.duration, min_embedding_duration)
        duration = min(duration, audio_duration)
        start = row.midpoint - duration / 2.0
        end = row.midpoint + duration / 2.0
        if start < 0:
            end -= start
            start = 0.0
        if end > audio_duration:
            start -= end - audio_duration
            end = audio_duration
            start = max(0.0, start)
        windows.append(EmbeddingWindow(row.row_index, round(start, 6), round(end, 6)))
    return windows


def build_tracklets(rows: list[SubtitleRow], merge_unknown_colors: bool = False) -> list[Tracklet]:
    if not rows:
        return []
    tracklets: list[Tracklet] = []
    start_position = 0
    current_key = color_key(rows[0].color)

    for position in range(1, len(rows) + 1):
        at_end = position == len(rows)
        next_key = None if at_end else color_key(rows[position].color)
        same_key = next_key == current_key and (current_key is not None or merge_unknown_colors)
        if not at_end and same_key:
            continue

        run_rows = rows[start_position:position]
        duration = sum(row.duration for row in run_rows)
        tracklets.append(
            Tracklet(
                tracklet_id=len(tracklets) + 1,
                row_indices=[row.row_index for row in run_rows],
                row_positions=list(range(start_position, position)),
                color=run_rows[0].color,
                normalized_color=current_key,
                start=run_rows[0].start,
                end=run_rows[-1].end,
                duration=duration,
            )
        )
        if not at_end:
            start_position = position
            current_key = next_key
    return tracklets


def build_cannot_links(tracklets: list[Tracklet]) -> set[tuple[int, int]]:
    cannot_links = set()
    for left, right in zip(tracklets, tracklets[1:]):
        if left.normalized_color != right.normalized_color:
            cannot_links.add(tuple(sorted((left.tracklet_id - 1, right.tracklet_id - 1))))
    return cannot_links


def tracklet_gap(left: Tracklet, right: Tracklet) -> float:
    return right.start - left.end


def local_color_window_score(
    color_counts: Counter[str],
    turn_count: int,
    span: float,
    min_turns: int,
) -> tuple[int, int, int, int, float] | None:
    if turn_count < min_turns:
        return None
    if len(color_counts) < 2:
        return None
    repeated_colors = sum(1 for count in color_counts.values() if count >= 2)
    if repeated_colors == 0:
        return None
    repeated_turns = sum(count - 1 for count in color_counts.values() if count >= 2)
    singleton_colors = sum(1 for count in color_counts.values() if count == 1)
    return (repeated_colors, repeated_turns, -singleton_colors, turn_count, -span)


def build_local_color_windows(
    tracklets: list[Tracklet],
    max_gap: float,
    max_span: float,
    max_colors: int,
    min_turns: int,
) -> list[LocalColorWindow]:
    if max_colors < 2 or min_turns < 2:
        return []

    windows: list[LocalColorWindow] = []
    start_index = 0
    while start_index < len(tracklets):
        first = tracklets[start_index]
        if first.normalized_color is None:
            start_index += 1
            continue

        color_counts: Counter[str] = Counter()
        colors_in_order: list[str] = []
        best_end: int | None = None
        best_score: tuple[int, int, int, int, float] | None = None
        best_max_gap = 0.0
        current_max_gap = 0.0

        for end_index in range(start_index, len(tracklets)):
            current = tracklets[end_index]
            if current.normalized_color is None:
                break
            if end_index > start_index:
                gap = tracklet_gap(tracklets[end_index - 1], current)
                if gap > max_gap:
                    break
                current_max_gap = max(current_max_gap, gap)

            span = current.end - first.start
            if span > max_span:
                break

            next_counts = color_counts.copy()
            next_counts[current.normalized_color] += 1
            if len(next_counts) > max_colors:
                break

            color_counts = next_counts
            if current.normalized_color not in colors_in_order:
                colors_in_order.append(current.normalized_color)

            score = local_color_window_score(
                color_counts,
                end_index - start_index + 1,
                span,
                min_turns,
            )
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_end = end_index
                best_max_gap = current_max_gap

        if best_end is None:
            start_index += 1
            continue

        window_tracklets = tracklets[start_index : best_end + 1]
        window_colors = []
        for tracklet in window_tracklets:
            if tracklet.normalized_color is not None and tracklet.normalized_color not in window_colors:
                window_colors.append(tracklet.normalized_color)
        windows.append(
            LocalColorWindow(
                window_id=len(windows) + 1,
                tracklet_indices=list(range(start_index, best_end + 1)),
                colors=window_colors,
                start=window_tracklets[0].start,
                end=window_tracklets[-1].end,
                span=window_tracklets[-1].end - window_tracklets[0].start,
                max_gap=best_max_gap,
            )
        )
        start_index = best_end + 1

    return windows


def build_color_role_units(
    tracklets: list[Tracklet],
    local_windows: list[LocalColorWindow],
    base_cannot_links: set[tuple[int, int]],
) -> tuple[list[ColorRoleUnit], list[int], set[tuple[int, int]], set[tuple[int, int]], dict[str, Any]]:
    union_find = UnionFind(len(tracklets))
    tracklet_window_ids: defaultdict[int, list[int]] = defaultdict(list)
    must_link_pairs: set[tuple[int, int]] = set()
    local_color_cannot_links: set[tuple[int, int]] = set()

    for window in local_windows:
        by_color: defaultdict[str, list[int]] = defaultdict(list)
        for index in window.tracklet_indices:
            tracklet_window_ids[index].append(window.window_id)
            color = tracklets[index].normalized_color
            if color is not None:
                by_color[color].append(index)

        for indices in by_color.values():
            if len(indices) < 2:
                continue
            first = indices[0]
            for index in indices[1:]:
                if union_find.union(first, index):
                    must_link_pairs.add(tuple(sorted((first, index))))

        for left_position, left in enumerate(window.tracklet_indices):
            for right in window.tracklet_indices[left_position + 1 :]:
                if tracklets[left].normalized_color != tracklets[right].normalized_color:
                    local_color_cannot_links.add(tuple(sorted((left, right))))

    components: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(tracklets)):
        components[union_find.find(index)].append(index)

    ordered_components = sorted(components.values(), key=lambda indices: min(indices))
    tracklet_to_unit = [-1] * len(tracklets)
    role_units: list[ColorRoleUnit] = []
    for unit_index, indices in enumerate(ordered_components):
        for index in indices:
            tracklet_to_unit[index] = unit_index
        colors = {tracklets[index].normalized_color for index in indices}
        role_units.append(
            ColorRoleUnit(
                unit_id=unit_index + 1,
                tracklet_indices=indices,
                normalized_color=colors.pop() if len(colors) == 1 else None,
                start=min(tracklets[index].start for index in indices),
                end=max(tracklets[index].end for index in indices),
                duration=sum(tracklets[index].duration for index in indices),
                row_count=sum(len(tracklets[index].row_indices) for index in indices),
                window_ids=sorted({window_id for index in indices for window_id in tracklet_window_ids[index]}),
            )
        )

    effective_tracklet_cannot_links = set(base_cannot_links) | local_color_cannot_links
    unit_cannot_links: set[tuple[int, int]] = set()
    for left, right in effective_tracklet_cannot_links:
        left_unit = tracklet_to_unit[left]
        right_unit = tracklet_to_unit[right]
        if left_unit != right_unit:
            unit_cannot_links.add(tuple(sorted((left_unit, right_unit))))

    collapsed_units = [unit for unit in role_units if len(unit.tracklet_indices) > 1]
    stats = {
        "enabled": bool(local_windows),
        "windows": len(local_windows),
        "role_units": len(role_units),
        "collapsed_role_units": len(collapsed_units),
        "tracklets_in_collapsed_units": sum(len(unit.tracklet_indices) for unit in collapsed_units),
        "must_link_pairs": len(must_link_pairs),
        "base_cannot_links": len(base_cannot_links),
        "local_color_cannot_links": len(local_color_cannot_links),
        "effective_tracklet_cannot_links": len(effective_tracklet_cannot_links),
        "unit_cannot_links": len(unit_cannot_links),
    }
    return role_units, tracklet_to_unit, effective_tracklet_cannot_links, unit_cannot_links, stats


def split_anchor_weak_items(
    durations: list[float],
    item_counts: list[int],
    anchor_min_duration: float,
    anchor_min_segments: int,
) -> tuple[list[int], list[int], int | None]:
    """Split clustering items into reliable anchors and assignment-only weak items.

    A short one-row unit is the noisiest case because its cluster evidence is
    just one short segment embedding. Multi-row or local-color-merged units are
    kept as anchors by default because their weighted average is more stable.
    """

    if len(durations) != len(item_counts):
        raise ValueError("durations and item_counts must have the same length")

    anchor_indices: list[int] = []
    weak_indices: list[int] = []
    for index, (duration, item_count) in enumerate(zip(durations, item_counts)):
        is_anchor = duration >= anchor_min_duration or item_count >= anchor_min_segments
        if is_anchor:
            anchor_indices.append(index)
        else:
            weak_indices.append(index)

    promoted_index = None
    if not anchor_indices and weak_indices:
        promoted_index = max(weak_indices, key=lambda index: (durations[index], -index))
        weak_indices = [index for index in weak_indices if index != promoted_index]
        anchor_indices = [promoted_index]

    return anchor_indices, weak_indices, promoted_index


def split_anchor_weak_tracklets(
    tracklets: list[Tracklet],
    anchor_min_duration: float,
    anchor_min_segments: int,
) -> tuple[list[int], list[int], int | None]:
    return split_anchor_weak_items(
        [tracklet.duration for tracklet in tracklets],
        [len(tracklet.row_indices) for tracklet in tracklets],
        anchor_min_duration,
        anchor_min_segments,
    )


def remap_cannot_links(
    cannot_links: set[tuple[int, int]],
    selected_indices: list[int],
) -> set[tuple[int, int]]:
    selected_map = {original_index: selected_index for selected_index, original_index in enumerate(selected_indices)}
    return {
        tuple(sorted((selected_map[left], selected_map[right])))
        for left, right in cannot_links
        if left in selected_map and right in selected_map
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, audio_path: Path, windows: list[EmbeddingWindow]) -> None:
    resolved_audio = str(audio_path.resolve())
    rows = [
        {
            "audio_filepath": resolved_audio,
            "offset": window.start,
            "duration": round(window.duration, 6),
            "label": "infer",
            "text": "-",
            "uniq_id": f"row_{window.row_index:06d}",
            "row_index": window.row_index,
        }
        for window in windows
    ]
    write_jsonl(path, rows)


def load_titanet_model(model_name: str, device: str):
    torch_mod = require_torch()
    import numpy as np

    if not hasattr(np, "sctypes"):
        np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64]}  # type: ignore[attr-defined]
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR is required for TitaNet embedding extraction. Install "
            "`nemo_toolkit[asr]` first."
        ) from exc

    map_location = torch_mod.device(device)
    if model_name.endswith(".nemo"):
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_name,
            map_location=map_location,
        )
    else:
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=model_name,
            map_location=map_location,
        )
    model = model.to(map_location)
    model.eval()
    return model


def l2_normalize(matrix: Any) -> Any:
    torch_mod = require_torch()
    matrix = matrix.float()
    if matrix.ndim == 1:
        norm = torch_mod.linalg.vector_norm(matrix)
        return matrix / norm if norm > 0 else matrix
    norms = torch_mod.linalg.vector_norm(matrix, dim=1, keepdim=True)
    norms = torch_mod.where(norms > 0, norms, torch_mod.ones_like(norms))
    return matrix / norms


def extract_segment_embeddings(
    model: Any,
    manifest_path: Path,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    expected_count: int,
    progress_every: int,
    progress: ProgressLogger | None = None,
) -> Any:
    torch_mod = require_torch()
    config = {
        "manifest_filepath": str(manifest_path),
        "sample_rate": sample_rate,
        "batch_size": batch_size,
        "trim_silence": False,
        "labels": None,
        "num_workers": num_workers,
    }
    model.setup_test_data(config)
    dataloader = model.test_dataloader()
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    batches = []
    rows_done = 0
    with torch_mod.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            moved = [item.to(model.device) if hasattr(item, "to") else item for item in batch]
            audio_signal = moved[0]
            audio_signal_len = moved[1]
            try:
                _logits, embeddings = model.forward(
                    input_signal=audio_signal,
                    input_signal_length=audio_signal_len,
                )
            except TypeError:
                _logits, embeddings = model.forward(input_signal=audio_signal, length=audio_signal_len)
            embeddings = embeddings.detach().float().cpu().reshape(-1, embeddings.shape[-1])
            batches.append(embeddings)
            rows_done += embeddings.shape[0]
            if progress and progress_every > 0:
                if batch_index == 1 or batch_index % progress_every == 0 or rows_done >= expected_count:
                    total_text = total_batches if total_batches is not None else "?"
                    progress.log(
                        f"embedded segment batch {batch_index}/{total_text}, "
                        f"rows {min(rows_done, expected_count)}/{expected_count}"
                    )

    if not batches:
        raise RuntimeError(f"No embeddings extracted from {manifest_path}")
    output = torch_mod.cat(batches, dim=0)
    if output.shape[0] != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {output.shape[0]}")
    return l2_normalize(output)


def weighted_tracklet_embeddings(rows: list[SubtitleRow], tracklets: list[Tracklet], segment_embeddings: Any) -> Any:
    torch_mod = require_torch()
    vectors = []
    for tracklet in tracklets:
        weights = torch_mod.tensor(
            [rows[position].duration for position in tracklet.row_positions],
            dtype=torch_mod.float32,
        )
        weights = weights / weights.sum()
        row_vectors = segment_embeddings[tracklet.row_positions]
        vectors.append((row_vectors * weights.unsqueeze(1)).sum(dim=0))
    return l2_normalize(torch_mod.stack(vectors, dim=0))


def weighted_color_role_unit_embeddings(
    tracklet_embeddings: Any,
    tracklets: list[Tracklet],
    role_units: list[ColorRoleUnit],
) -> Any:
    torch_mod = require_torch()
    vectors = []
    for unit in role_units:
        weights = torch_mod.tensor(
            [max(tracklets[index].duration, 1e-6) for index in unit.tracklet_indices],
            dtype=tracklet_embeddings.dtype,
            device=tracklet_embeddings.device,
        )
        weights = weights / weights.sum()
        vectors.append((tracklet_embeddings[unit.tracklet_indices] * weights.unsqueeze(1)).sum(dim=0))
    return l2_normalize(torch_mod.stack(vectors, dim=0))


def expand_unit_labels_to_tracklets(
    role_units: list[ColorRoleUnit],
    unit_labels: list[int],
    tracklet_count: int,
) -> list[int]:
    if len(role_units) != len(unit_labels):
        raise ValueError("role_units and unit_labels must have the same length")
    labels = [-1] * tracklet_count
    for unit, label in zip(role_units, unit_labels):
        for tracklet_index in unit.tracklet_indices:
            labels[tracklet_index] = label
    if any(label < 0 for label in labels):
        missing = [index for index, label in enumerate(labels) if label < 0]
        raise RuntimeError(f"Unassigned tracklet labels remain after unit expansion: {missing[:10]}")
    return labels


def expand_unit_assignment_kinds_to_tracklets(
    role_units: list[ColorRoleUnit],
    unit_assignment_kinds: list[str],
    tracklet_count: int,
) -> list[str]:
    if len(role_units) != len(unit_assignment_kinds):
        raise ValueError("role_units and unit_assignment_kinds must have the same length")
    assignment_kinds = ["unassigned"] * tracklet_count
    for unit, assignment_kind in zip(role_units, unit_assignment_kinds):
        if len(unit.tracklet_indices) > 1:
            assignment_kind = f"local_color_role_{assignment_kind}"
        for tracklet_index in unit.tracklet_indices:
            assignment_kinds[tracklet_index] = assignment_kind
    if any(kind == "unassigned" for kind in assignment_kinds):
        missing = [index for index, kind in enumerate(assignment_kinds) if kind == "unassigned"]
        raise RuntimeError(f"Unassigned tracklet assignment kinds remain: {missing[:10]}")
    return assignment_kinds


def cosine_distance(left: Any, right: Any) -> float:
    torch_mod = require_torch()
    return float(1.0 - torch_mod.dot(left, right))


def cluster_has_cannot_link(
    left_members: set[int],
    right_members: set[int],
    cannot_links: set[tuple[int, int]],
) -> bool:
    for left in left_members:
        for right in right_members:
            if tuple(sorted((left, right))) in cannot_links:
                return True
    return False


def build_spectral_affinity(
    embeddings: Any,
    cannot_links: set[tuple[int, int]],
    neighbors: int,
    sigma: float,
) -> Any:
    torch_mod = require_torch()
    embeddings = l2_normalize(embeddings)
    similarity = torch_mod.clamp(embeddings @ embeddings.T, min=-1.0, max=1.0)
    distance = torch_mod.clamp(1.0 - similarity, min=0.0)
    if sigma > 0:
        affinity = torch_mod.exp(-(distance**2) / (2.0 * sigma * sigma))
    else:
        affinity = torch_mod.clamp((similarity + 1.0) / 2.0, min=0.0)
    affinity.fill_diagonal_(0.0)
    for left, right in cannot_links:
        affinity[left, right] = 0.0
        affinity[right, left] = 0.0

    if neighbors > 0 and neighbors < affinity.shape[0] - 1:
        sparse = torch_mod.zeros_like(affinity)
        top_values, top_indices = torch_mod.topk(affinity, k=neighbors, dim=1)
        sparse.scatter_(1, top_indices, top_values)
        affinity = torch_mod.maximum(sparse, sparse.T)

    affinity.fill_diagonal_(0.0)
    return affinity


def normalized_laplacian(affinity: Any) -> Any:
    torch_mod = require_torch()
    degree = affinity.sum(dim=1)
    inv_sqrt_degree = torch_mod.where(
        degree > 1e-12,
        1.0 / torch_mod.sqrt(degree),
        torch_mod.zeros_like(degree),
    )
    identity = torch_mod.eye(affinity.shape[0], dtype=affinity.dtype, device=affinity.device)
    return identity - inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :]


def estimate_speaker_count_from_eigengap(eigenvalues: Any, min_speakers: int, max_speakers: int) -> tuple[int, float]:
    eigen_count = int(eigenvalues.shape[0])
    if eigen_count <= 1:
        return 1, 0.0
    max_candidate = max(1, min(max_speakers, eigen_count - 1))
    min_k = max(1, min(min_speakers, max_candidate))
    max_k = max(min_k, max_candidate)
    best_k = min_k
    best_gap = -1.0
    for k in range(min_k, max_k + 1):
        gap = float(eigenvalues[k] - eigenvalues[k - 1])
        if gap > best_gap:
            best_gap = gap
            best_k = k
    return best_k, best_gap


def deterministic_kmeans(data: Any, cluster_count: int, max_iters: int) -> tuple[list[int], int]:
    torch_mod = require_torch()
    row_count = int(data.shape[0])
    cluster_count = max(1, min(cluster_count, row_count))
    if cluster_count == 1:
        return [0] * row_count, 0

    # Farthest-point initialization is deterministic and avoids random seeds.
    centroids = []
    first_index = int(torch_mod.argmax(torch_mod.linalg.vector_norm(data, dim=1)).item())
    centroids.append(data[first_index])
    min_distances = torch_mod.sum((data - centroids[0]) ** 2, dim=1)
    for _ in range(1, cluster_count):
        next_index = int(torch_mod.argmax(min_distances).item())
        centroids.append(data[next_index])
        distances = torch_mod.sum((data - centroids[-1]) ** 2, dim=1)
        min_distances = torch_mod.minimum(min_distances, distances)
    centroid_tensor = torch_mod.stack(centroids, dim=0)

    labels = torch_mod.zeros(row_count, dtype=torch_mod.long)
    iterations = 0
    for iteration in range(max_iters):
        distances = torch_mod.cdist(data, centroid_tensor, p=2.0) ** 2
        new_labels = torch_mod.argmin(distances, dim=1)
        if iteration > 0 and torch_mod.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(cluster_count):
            mask = labels == cluster_id
            if bool(mask.any()):
                centroid_tensor[cluster_id] = data[mask].mean(dim=0)
            else:
                farthest = int(torch_mod.argmax(torch_mod.min(distances, dim=1).values).item())
                centroid_tensor[cluster_id] = data[farthest]
                labels[farthest] = cluster_id
        iterations = iteration + 1
    return [int(label) for label in labels.tolist()], iterations


def compact_labels(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    compacted = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        compacted.append(mapping[label])
    return compacted


def label_centroids(embeddings: Any, labels: list[int]) -> dict[int, Any]:
    torch_mod = require_torch()
    label_tensor = torch_mod.tensor(labels, dtype=torch_mod.long)
    centroids = {}
    for label in sorted(set(labels)):
        mask = label_tensor == label
        centroids[label] = l2_normalize(embeddings[mask].mean(dim=0))
    return centroids


def weighted_label_centroids(
    embeddings: Any,
    labels: list[int],
    durations: list[float],
    indices: list[int],
) -> dict[int, Any]:
    torch_mod = require_torch()
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        label = labels[index]
        if label >= 0:
            grouped_indices[label].append(index)

    centroids = {}
    for label, member_indices in sorted(grouped_indices.items()):
        weights = torch_mod.tensor(
            [max(durations[index], 1e-6) for index in member_indices],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        vectors = embeddings[member_indices]
        centroid = (vectors * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        centroids[label] = l2_normalize(centroid)
    return centroids


def violates_label(index: int, candidate_label: int, labels: list[int], cannot_links: set[tuple[int, int]]) -> bool:
    for left, right in cannot_links:
        other = None
        if left == index:
            other = right
        elif right == index:
            other = left
        if other is not None and labels[other] == candidate_label:
            return True
    return False


def assign_weak_tracklets_to_anchor_centroids(
    embeddings: Any,
    durations: list[float],
    cannot_links: set[tuple[int, int]],
    anchor_indices: list[int],
    anchor_labels: list[int],
    weak_indices: list[int],
    weak_assignment_threshold: float,
    promoted_anchor_index: int | None = None,
) -> tuple[list[int], list[str], dict[str, Any]]:
    torch_mod = require_torch()
    if len(anchor_indices) != len(anchor_labels):
        raise ValueError("anchor_indices and anchor_labels must have the same length")
    if not anchor_indices and weak_indices:
        raise ValueError("At least one anchor is required before assigning weak tracklets")

    labels = [-1] * len(durations)
    assignment_kinds = ["weak_tracklet_unassigned"] * len(durations)
    for original_index, label in zip(anchor_indices, anchor_labels):
        labels[original_index] = label
        assignment_kinds[original_index] = (
            "anchor_tracklet_promoted" if original_index == promoted_anchor_index else "anchor_tracklet_clustered"
        )

    centroids = weighted_label_centroids(embeddings, labels, durations, anchor_indices)
    next_label = max(labels) + 1 if any(label >= 0 for label in labels) else 0
    weak_assigned = 0
    weak_singletons = 0
    weak_assignment_distances: list[float] = []

    for weak_index in sorted(weak_indices):
        candidates = []
        for label, centroid in centroids.items():
            if violates_label(weak_index, label, labels, cannot_links):
                continue
            similarity = float(torch_mod.dot(embeddings[weak_index], centroid))
            distance = 1.0 - similarity
            candidates.append((distance, label, similarity))
        candidates.sort(key=lambda item: (item[0], item[1]))

        should_create_singleton = not candidates
        if candidates and weak_assignment_threshold >= 0:
            should_create_singleton = candidates[0][0] > weak_assignment_threshold

        if should_create_singleton:
            labels[weak_index] = next_label
            next_label += 1
            weak_singletons += 1
            assignment_kinds[weak_index] = "weak_tracklet_singleton"
        else:
            best_distance, best_label, _best_similarity = candidates[0]
            labels[weak_index] = best_label
            weak_assigned += 1
            weak_assignment_distances.append(best_distance)
            assignment_kinds[weak_index] = "weak_tracklet_centroid_assignment"

    if any(label < 0 for label in labels):
        missing = [index for index, label in enumerate(labels) if label < 0]
        raise RuntimeError(f"Unassigned tracklet labels remain: {missing[:10]}")

    final_violations = sum(1 for left, right in cannot_links if labels[left] == labels[right])
    compacted_labels = compact_labels(labels)
    assignment_counts = {
        kind: sum(1 for assignment_kind in assignment_kinds if assignment_kind == kind)
        for kind in sorted(set(assignment_kinds))
    }
    stats = {
        "short_tracklet_mode": "assign-after",
        "anchors": len(anchor_indices),
        "weak_tracklets": len(weak_indices),
        "weak_assigned_to_anchor_centroid": weak_assigned,
        "weak_singletons": weak_singletons,
        "weak_assignment_threshold": weak_assignment_threshold,
        "promoted_anchor_tracklet_id": promoted_anchor_index + 1 if promoted_anchor_index is not None else None,
        "assignment_counts": assignment_counts,
        "weak_assignment_mean_distance": (
            round(sum(weak_assignment_distances) / len(weak_assignment_distances), 8)
            if weak_assignment_distances
            else None
        ),
        "weak_assignment_max_distance": (
            round(max(weak_assignment_distances), 8) if weak_assignment_distances else None
        ),
        "final_cannot_link_violations_after_weak_assignment": final_violations,
    }
    return compacted_labels, assignment_kinds, stats


def repair_cannot_link_violations(
    embeddings: Any,
    labels: list[int],
    durations: list[float],
    cannot_links: set[tuple[int, int]],
    max_passes: int = 10,
) -> tuple[list[int], int, int]:
    torch_mod = require_torch()
    repaired = list(labels)
    repairs = 0

    for _pass_index in range(max_passes):
        violations = [(left, right) for left, right in cannot_links if repaired[left] == repaired[right]]
        if not violations:
            break
        changed = False
        centroids = label_centroids(embeddings, repaired)
        existing_labels = sorted(centroids)
        for left, right in violations:
            candidates_to_move = sorted((left, right), key=lambda index: (durations[index], index))
            moved = False
            for index in candidates_to_move:
                current_label = repaired[index]
                allowed_labels = [
                    label
                    for label in existing_labels
                    if label != current_label and not violates_label(index, label, repaired, cannot_links)
                ]
                if allowed_labels:
                    scores = [
                        (float(torch_mod.dot(embeddings[index], centroids[label])), label)
                        for label in allowed_labels
                    ]
                    scores.sort(reverse=True)
                    repaired[index] = scores[0][1]
                else:
                    new_label = max(repaired) + 1
                    repaired[index] = new_label
                    existing_labels.append(new_label)
                    centroids[new_label] = embeddings[index]
                repairs += 1
                changed = True
                moved = True
                break
            if moved:
                continue
        if not changed:
            break

    final_violations = sum(1 for left, right in cannot_links if repaired[left] == repaired[right])
    return compact_labels(repaired), repairs, final_violations


def constrained_spectral_clustering(
    embeddings: Any,
    durations: list[float],
    cannot_links: set[tuple[int, int]],
    num_speakers: int | None,
    min_num_speakers: int,
    max_num_speakers: int,
    spectral_neighbors: int,
    spectral_sigma: float,
    kmeans_iters: int,
    progress: ProgressLogger | None = None,
) -> tuple[list[int], dict[str, Any]]:
    torch_mod = require_torch()
    row_count = int(embeddings.shape[0])
    if row_count == 0:
        return [], {"mode": "constrained_spectral", "clusters": 0}
    if row_count == 1:
        return [0], {"mode": "constrained_spectral", "clusters": 1}

    if progress:
        progress.log(
            f"building spectral affinity graph: tracklets={row_count}, "
            f"neighbors={spectral_neighbors}, cannot_links={len(cannot_links)}"
        )
    affinity = build_spectral_affinity(embeddings, cannot_links, spectral_neighbors, spectral_sigma)
    laplacian = normalized_laplacian(affinity).cpu()
    eigenvalues, eigenvectors = torch_mod.linalg.eigh(laplacian)

    if num_speakers is None:
        cluster_count, eigengap = estimate_speaker_count_from_eigengap(
            eigenvalues,
            min_num_speakers,
            min(max_num_speakers, row_count),
        )
        speaker_count_mode = "eigengap"
    else:
        cluster_count = max(1, min(num_speakers, row_count))
        eigengap = None
        speaker_count_mode = "oracle"

    if progress:
        progress.log(f"spectral clustering with k={cluster_count} ({speaker_count_mode})")
    spectral_embedding = eigenvectors[:, :cluster_count]
    spectral_embedding = l2_normalize(spectral_embedding)
    labels, iterations = deterministic_kmeans(spectral_embedding, cluster_count, kmeans_iters)
    initial_violations = sum(1 for left, right in cannot_links if labels[left] == labels[right])
    labels, repairs, final_violations = repair_cannot_link_violations(
        embeddings.cpu(),
        labels,
        durations,
        cannot_links,
    )

    stats = {
        "mode": "constrained_spectral",
        "num_speakers_requested": num_speakers,
        "speaker_count_mode": speaker_count_mode,
        "min_num_speakers": min_num_speakers,
        "max_num_speakers": max_num_speakers,
        "selected_clusters_before_repair": cluster_count,
        "clusters": len(set(labels)),
        "spectral_neighbors": spectral_neighbors,
        "spectral_sigma": spectral_sigma,
        "kmeans_iterations": iterations,
        "cannot_links": len(cannot_links),
        "initial_cannot_link_violations": initial_violations,
        "cannot_link_repairs": repairs,
        "final_cannot_link_violations": final_violations,
        "eigengap": round(eigengap, 8) if eigengap is not None else None,
        "first_eigenvalues": [round(float(value), 8) for value in eigenvalues[: min(20, len(eigenvalues))]],
    }
    return labels, stats


def constrained_agglomerative_clustering(
    embeddings: Any,
    durations: list[float],
    cannot_links: set[tuple[int, int]],
    num_speakers: int | None,
    max_num_speakers: int,
    distance_threshold: float,
    progress: ProgressLogger | None = None,
) -> tuple[list[int], dict[str, Any]]:
    torch_mod = require_torch()
    cluster_members: dict[int, set[int]] = {index: {index} for index in range(len(durations))}
    centroids: dict[int, Any] = {index: embeddings[index].clone() for index in range(len(durations))}
    cluster_durations: dict[int, float] = {index: durations[index] for index in range(len(durations))}
    next_cluster_id = len(durations)
    merges = 0
    blocked_pairs_seen = 0

    def target_reached() -> bool:
        if num_speakers is not None:
            return len(cluster_members) <= num_speakers
        return len(cluster_members) <= max_num_speakers

    while len(cluster_members) > 1:
        ids = sorted(cluster_members)
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        blocked_this_round = 0
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1 :]:
                if cluster_has_cannot_link(cluster_members[left_id], cluster_members[right_id], cannot_links):
                    blocked_this_round += 1
                    continue
                distance = cosine_distance(centroids[left_id], centroids[right_id])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left_id, right_id)
        blocked_pairs_seen += blocked_this_round
        if best_pair is None:
            break
        if num_speakers is None and len(cluster_members) <= max_num_speakers and best_distance > distance_threshold:
            break

        left_id, right_id = best_pair
        left_weight = cluster_durations[left_id]
        right_weight = cluster_durations[right_id]
        merged_duration = left_weight + right_weight
        merged_centroid = (
            centroids[left_id] * left_weight + centroids[right_id] * right_weight
        ) / merged_duration
        merged_centroid = l2_normalize(merged_centroid)
        merged_members = cluster_members[left_id] | cluster_members[right_id]
        del cluster_members[left_id], cluster_members[right_id]
        del centroids[left_id], centroids[right_id]
        del cluster_durations[left_id], cluster_durations[right_id]
        cluster_members[next_cluster_id] = merged_members
        centroids[next_cluster_id] = merged_centroid
        cluster_durations[next_cluster_id] = merged_duration
        next_cluster_id += 1
        merges += 1
        if progress and merges % 100 == 0:
            progress.log(f"clustering merges={merges}, active_clusters={len(cluster_members)}")
        if target_reached() and num_speakers is not None:
            break

    labels = [-1] * len(durations)
    ordered_clusters = sorted(
        cluster_members.items(),
        key=lambda item: (min(item[1]), -sum(durations[index] for index in item[1])),
    )
    for label, (_cluster_id, members) in enumerate(ordered_clusters):
        for member in members:
            labels[member] = label

    stats = {
        "mode": "constrained_agglomerative",
        "num_speakers_requested": num_speakers,
        "max_num_speakers": max_num_speakers,
        "distance_threshold": distance_threshold,
        "merges": merges,
        "clusters": len(set(labels)),
        "cannot_links": len(cannot_links),
        "blocked_pairs_seen": blocked_pairs_seen,
        "stopped_above_requested_speakers": (
            num_speakers is not None and len(set(labels)) > num_speakers
        ),
    }
    return labels, stats


def stable_speaker_ids(tracklets: list[Tracklet], labels: list[int]) -> dict[int, int]:
    sortable = []
    for cluster_id in sorted(set(labels)):
        members = [tracklet for tracklet, label in zip(tracklets, labels) if label == cluster_id]
        sortable.append((min(tracklet.start for tracklet in members), -sum(t.duration for t in members), cluster_id))
    sortable.sort()
    return {cluster_id: speaker_id for speaker_id, (_start, _duration, cluster_id) in enumerate(sortable, 1)}


def rttm_line(file_id: str, start: float, duration: float, speaker_label: str) -> str:
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>"


def write_outputs(
    output_dir: Path,
    output_rttm: Path,
    rows: list[SubtitleRow],
    windows: list[EmbeddingWindow],
    tracklets: list[Tracklet],
    tracklet_labels: list[int],
    speaker_ids: dict[int, int],
    cannot_links: set[tuple[int, int]],
    clustering_stats: dict[str, Any],
    model_name: str,
    file_id: str,
    tracklet_assignment_kinds: list[str] | None = None,
    local_color_windows: list[LocalColorWindow] | None = None,
    color_role_units: list[ColorRoleUnit] | None = None,
    tracklet_to_role_unit: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tracklet_assignment_kinds is None:
        tracklet_assignment_kinds = ["titanet_tracklet_clustered"] * len(tracklets)
    if len(tracklet_assignment_kinds) != len(tracklets):
        raise ValueError("tracklet_assignment_kinds must match tracklets length")
    if tracklet_to_role_unit is not None and len(tracklet_to_role_unit) != len(tracklets):
        raise ValueError("tracklet_to_role_unit must match tracklets length")

    tracklet_by_position = {}
    for tracklet_index, (tracklet, cluster_id, assignment_kind) in enumerate(
        zip(tracklets, tracklet_labels, tracklet_assignment_kinds)
    ):
        role_unit_id = tracklet_to_role_unit[tracklet_index] + 1 if tracklet_to_role_unit is not None else None
        role_window_ids = (
            color_role_units[tracklet_to_role_unit[tracklet_index]].window_ids
            if color_role_units is not None and tracklet_to_role_unit is not None
            else []
        )
        for position in tracklet.row_positions:
            tracklet_by_position[position] = (tracklet, cluster_id, assignment_kind, role_unit_id, role_window_ids)

    line_rows = []
    rttm_rows = []
    for position, row in enumerate(rows):
        tracklet, cluster_id, assignment_kind, role_unit_id, role_window_ids = tracklet_by_position[position]
        speaker_id = speaker_ids[cluster_id]
        speaker_label = f"speaker_{speaker_id:03d}"
        line_rows.append(
            {
                "unit_id": row.row_index,
                "row_index": row.row_index,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "cluster_id": cluster_id,
                "tracklet_id": tracklet.tracklet_id,
                "assignment_kind": assignment_kind,
                "local_color_role_unit_id": role_unit_id,
                "local_color_window_ids": role_window_ids,
                "start": row.start,
                "end": row.end,
                "duration": row.duration,
                "embedding_start": windows[position].start,
                "embedding_end": windows[position].end,
                "color": row.color,
                "text": row.text,
                "gold_speaker": row.gold_speaker,
                "color_cue_confidence": row.color_cue_confidence,
                "color_cue_ambiguous": row.color_cue_ambiguous,
            }
        )
        rttm_rows.append(rttm_line(file_id, row.start, row.duration, speaker_label))

    tracklet_rows = []
    for tracklet_index, (tracklet, cluster_id, assignment_kind) in enumerate(
        zip(tracklets, tracklet_labels, tracklet_assignment_kinds)
    ):
        speaker_id = speaker_ids[cluster_id]
        role_unit_id = tracklet_to_role_unit[tracklet_index] + 1 if tracklet_to_role_unit is not None else None
        role_window_ids = (
            color_role_units[tracklet_to_role_unit[tracklet_index]].window_ids
            if color_role_units is not None and tracklet_to_role_unit is not None
            else []
        )
        tracklet_rows.append(
            {
                "tracklet_id": tracklet.tracklet_id,
                "speaker_id": speaker_id,
                "speaker_label": f"speaker_{speaker_id:03d}",
                "cluster_id": cluster_id,
                "row_indices": tracklet.row_indices,
                "start": tracklet.start,
                "end": tracklet.end,
                "duration": tracklet.duration,
                "color": tracklet.color,
                "normalized_color": tracklet.normalized_color,
                "segment_count": len(tracklet.row_indices),
                "assignment_kind": assignment_kind,
                "local_color_role_unit_id": role_unit_id,
                "local_color_window_ids": role_window_ids,
            }
        )

    cluster_rows = []
    for cluster_id, speaker_id in sorted(speaker_ids.items(), key=lambda item: item[1]):
        members = [tracklet for tracklet, label in zip(tracklets, tracklet_labels) if label == cluster_id]
        cluster_rows.append(
            {
                "speaker_id": speaker_id,
                "speaker_label": f"speaker_{speaker_id:03d}",
                "cluster_id": cluster_id,
                "tracklet_count": len(members),
                "segment_count": sum(len(tracklet.row_indices) for tracklet in members),
                "total_duration": round(sum(tracklet.duration for tracklet in members), 6),
                "colors": sorted({tracklet.color for tracklet in members}),
                "tracklet_ids": [tracklet.tracklet_id for tracklet in members],
            }
        )

    cannot_link_rows = [
        {
            "left_tracklet_id": left + 1,
            "right_tracklet_id": right + 1,
            "left_color": tracklets[left].color,
            "right_color": tracklets[right].color,
        }
        for left, right in sorted(cannot_links)
    ]
    summary = {
        "pipeline": "titanet_tracklet_constrained",
        "speaker_model": model_name,
        "segments": len(rows),
        "tracklets": len(tracklets),
        "clusters": len(speaker_ids),
        "rttm_turns": len(rttm_rows),
        "clustering": clustering_stats,
    }

    write_jsonl(output_dir / "speaker_lines.jsonl", line_rows)
    write_jsonl(output_dir / "speaker_segments.jsonl", line_rows)
    write_jsonl(output_dir / "tracklets.jsonl", tracklet_rows)
    write_jsonl(output_dir / "cannot_links.jsonl", cannot_link_rows)
    if local_color_windows is not None:
        write_jsonl(
            output_dir / "local_color_windows.jsonl",
            [
                {
                    "window_id": window.window_id,
                    "tracklet_ids": [index + 1 for index in window.tracklet_indices],
                    "colors": window.colors,
                    "start": window.start,
                    "end": window.end,
                    "span": window.span,
                    "max_gap": window.max_gap,
                    "tracklet_count": len(window.tracklet_indices),
                }
                for window in local_color_windows
            ],
        )
    if color_role_units is not None:
        write_jsonl(
            output_dir / "local_color_role_units.jsonl",
            [
                {
                    "local_color_role_unit_id": unit.unit_id,
                    "tracklet_ids": [index + 1 for index in unit.tracklet_indices],
                    "normalized_color": unit.normalized_color,
                    "start": unit.start,
                    "end": unit.end,
                    "duration": unit.duration,
                    "row_count": unit.row_count,
                    "window_ids": unit.window_ids,
                }
                for unit in color_role_units
            ],
        )
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
    output_rttm.parent.mkdir(parents=True, exist_ok=True)
    output_rttm.write_text(rttm_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    progress = ProgressLogger()
    progress.log("starting TitaNet tracklet diarization pipeline")
    rows = parse_subtitle_rows(args.subtitle, max_rows=args.max_rows)
    if not rows:
        raise ValueError(f"No subtitle rows found in {args.subtitle}")
    progress.log(f"loaded {len(rows)} subtitle rows")
    tracklets = build_tracklets(rows, merge_unknown_colors=args.merge_unknown_colors)
    base_cannot_links = set() if args.disable_cannot_link else build_cannot_links(tracklets)
    progress.log(f"built {len(tracklets)} tracklets and {len(base_cannot_links)} adjacent cannot-link constraints")
    if args.disable_local_color_windows:
        local_color_windows: list[LocalColorWindow] = []
    else:
        local_color_windows = build_local_color_windows(
            tracklets,
            args.local_color_window_max_gap,
            args.local_color_window_max_span,
            args.local_color_window_max_colors,
            args.local_color_window_min_turns,
        )
    (
        color_role_units,
        tracklet_to_role_unit,
        effective_tracklet_cannot_links,
        unit_cannot_links,
        local_color_stats,
    ) = build_color_role_units(tracklets, local_color_windows, base_cannot_links)
    local_color_stats.update(
        {
            "disabled": args.disable_local_color_windows,
            "max_gap": args.local_color_window_max_gap,
            "max_span": args.local_color_window_max_span,
            "max_colors": args.local_color_window_max_colors,
            "min_turns": args.local_color_window_min_turns,
        }
    )
    progress.log(
        f"local color windows={len(local_color_windows)}, "
        f"clustering_units={len(color_role_units)}, unit_cannot_links={len(unit_cannot_links)}"
    )
    dry_run_anchor_indices: list[int] = []
    dry_run_weak_indices: list[int] = []
    dry_run_promoted_index: int | None = None
    role_unit_durations = [unit.duration for unit in color_role_units]
    role_unit_item_counts = [unit.row_count for unit in color_role_units]
    if args.short_tracklet_mode == "assign-after":
        dry_run_anchor_indices, dry_run_weak_indices, dry_run_promoted_index = split_anchor_weak_items(
            role_unit_durations,
            role_unit_item_counts,
            args.anchor_min_duration,
            args.anchor_min_segments,
        )
        progress.log(
            f"anchor split: anchor_units={len(dry_run_anchor_indices)}, "
            f"weak_units={len(dry_run_weak_indices)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run_anchor_set = set(dry_run_anchor_indices)
        dry_run_weak_set = set(dry_run_weak_indices)
        unit_role_by_tracklet = {}
        for tracklet_index, unit_index in enumerate(tracklet_to_role_unit):
            unit_role_by_tracklet[tracklet_index] = (
                "anchor_promoted"
                if unit_index == dry_run_promoted_index
                else "anchor"
                if unit_index in dry_run_anchor_set
                else "weak"
                if unit_index in dry_run_weak_set
                else "clustered"
            )
        write_jsonl(
            args.output_dir / "tracklets.jsonl",
            [
                {
                    "tracklet_id": tracklet.tracklet_id,
                    "row_indices": tracklet.row_indices,
                    "start": tracklet.start,
                    "end": tracklet.end,
                    "duration": tracklet.duration,
                    "color": tracklet.color,
                    "normalized_color": tracklet.normalized_color,
                    "local_color_role_unit_id": tracklet_to_role_unit[index] + 1,
                    "local_color_window_ids": color_role_units[tracklet_to_role_unit[index]].window_ids,
                    "assignment_role": unit_role_by_tracklet[index],
                }
                for index, tracklet in enumerate(tracklets)
            ],
        )
        write_jsonl(
            args.output_dir / "local_color_windows.jsonl",
            [
                {
                    "window_id": window.window_id,
                    "tracklet_ids": [index + 1 for index in window.tracklet_indices],
                    "colors": window.colors,
                    "start": window.start,
                    "end": window.end,
                    "span": window.span,
                    "max_gap": window.max_gap,
                    "tracklet_count": len(window.tracklet_indices),
                }
                for window in local_color_windows
            ],
        )
        write_jsonl(
            args.output_dir / "local_color_role_units.jsonl",
            [
                {
                    "local_color_role_unit_id": unit.unit_id,
                    "tracklet_ids": [index + 1 for index in unit.tracklet_indices],
                    "normalized_color": unit.normalized_color,
                    "start": unit.start,
                    "end": unit.end,
                    "duration": unit.duration,
                    "row_count": unit.row_count,
                    "window_ids": unit.window_ids,
                }
                for unit in color_role_units
            ],
        )
        print(f"rows: {len(rows)}")
        print(f"tracklets: {len(tracklets)}")
        print(f"adjacent_cannot_links: {len(base_cannot_links)}")
        print(f"local_color_windows: {len(local_color_windows)}")
        print(f"clustering_units: {len(color_role_units)}")
        print(f"unit_cannot_links: {len(unit_cannot_links)}")
        if args.short_tracklet_mode == "assign-after":
            print(f"anchor_units: {len(dry_run_anchor_indices)}")
            print(f"weak_units: {len(dry_run_weak_indices)}")
        print("Dry run complete; no audio, embeddings, clustering, or RTTM were generated.")
        return

    device = choose_device(args.device)
    progress.log(f"using device: {device}")
    audio_cache = args.audio_cache or args.embedding_dir / f"audio_{args.sample_rate}hz_mono.wav"
    extract_audio(args.video, audio_cache, args.sample_rate, args.overwrite_audio, progress=progress)
    audio_duration = wav_duration(audio_cache)
    windows = build_embedding_windows(rows, audio_duration, args.min_embedding_duration)
    manifest_path = args.embedding_dir / "segment_manifest.json"
    write_manifest(manifest_path, audio_cache, windows)
    progress.log(f"wrote segment manifest: {manifest_path}")

    progress.log(f"loading TitaNet speaker model: {args.speaker_model}")
    model = load_titanet_model(args.speaker_model, device)
    progress.log("speaker model ready")
    segment_embeddings = extract_segment_embeddings(
        model,
        manifest_path,
        args.sample_rate,
        args.batch_size,
        args.num_workers,
        len(rows),
        args.progress_every,
        progress=progress,
    )
    tracklet_embeddings = weighted_tracklet_embeddings(rows, tracklets, segment_embeddings)
    role_unit_embeddings = weighted_color_role_unit_embeddings(
        tracklet_embeddings,
        tracklets,
        color_role_units,
    )
    torch_mod = require_torch()
    torch_mod.save(
        {
            "segment_embeddings": segment_embeddings,
            "tracklet_embeddings": tracklet_embeddings,
            "local_color_role_unit_embeddings": role_unit_embeddings,
            "tracklets": [tracklet.__dict__ for tracklet in tracklets],
            "local_color_windows": [window.__dict__ for window in local_color_windows],
            "local_color_role_units": [unit.__dict__ for unit in color_role_units],
            "tracklet_to_role_unit": tracklet_to_role_unit,
            "embedding_windows": [window.__dict__ for window in windows],
            "speaker_model": args.speaker_model,
        },
        args.embedding_dir / "tracklet_embeddings.pt",
    )
    progress.log(f"saved embeddings: {args.embedding_dir / 'tracklet_embeddings.pt'}")

    tracklet_durations = [tracklet.duration for tracklet in tracklets]
    tracklet_assignment_kinds: list[str] | None = None
    if args.short_tracklet_mode == "assign-after":
        anchor_indices, weak_indices, promoted_index = split_anchor_weak_items(
            role_unit_durations,
            role_unit_item_counts,
            args.anchor_min_duration,
            args.anchor_min_segments,
        )
        anchor_cannot_links = remap_cannot_links(unit_cannot_links, anchor_indices)
        anchor_embeddings = role_unit_embeddings[anchor_indices]
        anchor_durations = [role_unit_durations[index] for index in anchor_indices]
        progress.log(
            f"clustering anchor color-role units with method: {args.clustering_method}; "
            f"anchor_units={len(anchor_indices)}, weak_units={len(weak_indices)}"
        )
        if args.clustering_method == "constrained-spectral":
            anchor_labels, anchor_clustering_stats = constrained_spectral_clustering(
                anchor_embeddings,
                anchor_durations,
                anchor_cannot_links,
                args.num_speakers,
                args.min_num_speakers,
                args.max_num_speakers,
                args.spectral_neighbors,
                args.spectral_sigma,
                args.spectral_kmeans_iters,
                progress=progress,
            )
        else:
            anchor_labels, anchor_clustering_stats = constrained_agglomerative_clustering(
                anchor_embeddings,
                anchor_durations,
                anchor_cannot_links,
                args.num_speakers,
                args.max_num_speakers,
                args.distance_threshold,
                progress=progress,
            )
        progress.log("assigning weak color-role units to fixed anchor centroids")
        role_unit_labels, unit_assignment_kinds, assignment_stats = assign_weak_tracklets_to_anchor_centroids(
            role_unit_embeddings,
            role_unit_durations,
            unit_cannot_links,
            anchor_indices,
            anchor_labels,
            weak_indices,
            args.weak_assignment_threshold,
            promoted_anchor_index=promoted_index,
        )
        tracklet_labels = expand_unit_labels_to_tracklets(
            color_role_units,
            role_unit_labels,
            len(tracklets),
        )
        tracklet_assignment_kinds = expand_unit_assignment_kinds_to_tracklets(
            color_role_units,
            unit_assignment_kinds,
            len(tracklets),
        )
        clustering_stats = {
            "mode": f"{args.clustering_method}_anchor_then_assign_weak",
            "base_clustering_method": args.clustering_method,
            "short_tracklet_mode": args.short_tracklet_mode,
            "anchor_min_duration": args.anchor_min_duration,
            "anchor_min_segments": args.anchor_min_segments,
            "tracklets": len(tracklets),
            "clustering_units": len(color_role_units),
            "anchor_units": len(anchor_indices),
            "weak_units": len(weak_indices),
            "anchors": len(anchor_indices),
            "weak_tracklets": len(weak_indices),
            "effective_tracklet_cannot_links": len(effective_tracklet_cannot_links),
            "unit_cannot_links": len(unit_cannot_links),
            "anchor_cannot_links": len(anchor_cannot_links),
            "clusters": len(set(tracklet_labels)),
            "unit_clusters": len(set(role_unit_labels)),
            "local_color_windows": local_color_stats,
            "anchor_clustering": anchor_clustering_stats,
            "weak_assignment": assignment_stats,
        }
    else:
        progress.log(f"clustering all color-role units with method: {args.clustering_method}")
        if args.clustering_method == "constrained-spectral":
            role_unit_labels, clustering_stats = constrained_spectral_clustering(
                role_unit_embeddings,
                role_unit_durations,
                unit_cannot_links,
                args.num_speakers,
                args.min_num_speakers,
                args.max_num_speakers,
                args.spectral_neighbors,
                args.spectral_sigma,
                args.spectral_kmeans_iters,
                progress=progress,
            )
        else:
            role_unit_labels, clustering_stats = constrained_agglomerative_clustering(
                role_unit_embeddings,
                role_unit_durations,
                unit_cannot_links,
                args.num_speakers,
                args.max_num_speakers,
                args.distance_threshold,
                progress=progress,
            )
        tracklet_labels = expand_unit_labels_to_tracklets(
            color_role_units,
            role_unit_labels,
            len(tracklets),
        )
        clustering_stats["short_tracklet_mode"] = args.short_tracklet_mode
        clustering_stats["tracklets"] = len(tracklets)
        clustering_stats["clustering_units"] = len(color_role_units)
        clustering_stats["effective_tracklet_cannot_links"] = len(effective_tracklet_cannot_links)
        clustering_stats["unit_cannot_links"] = len(unit_cannot_links)
        clustering_stats["unit_clusters"] = len(set(role_unit_labels))
        clustering_stats["local_color_windows"] = local_color_stats
        unit_assignment_kinds = ["titanet_tracklet_clustered"] * len(color_role_units)
        tracklet_assignment_kinds = expand_unit_assignment_kinds_to_tracklets(
            color_role_units,
            unit_assignment_kinds,
            len(tracklets),
        )

    speaker_ids = stable_speaker_ids(tracklets, tracklet_labels)
    write_outputs(
        args.output_dir,
        args.output_rttm,
        rows,
        windows,
        tracklets,
        tracklet_labels,
        speaker_ids,
        effective_tracklet_cannot_links,
        clustering_stats,
        args.speaker_model,
        args.rttm_file_id,
        tracklet_assignment_kinds=tracklet_assignment_kinds,
        local_color_windows=local_color_windows,
        color_role_units=color_role_units,
        tracklet_to_role_unit=tracklet_to_role_unit,
    )
    progress.log(f"final RTTM written: {args.output_rttm}")
    print(f"rows: {len(rows)}")
    print(f"tracklets: {len(tracklets)}")
    print(f"local_color_windows: {len(local_color_windows)}")
    print(f"clustering_units: {len(color_role_units)}")
    print(f"effective_tracklet_cannot_links: {len(effective_tracklet_cannot_links)}")
    print(f"unit_cannot_links: {len(unit_cannot_links)}")
    if args.short_tracklet_mode == "assign-after":
        print(f"anchor_units: {clustering_stats['anchor_units']}")
        print(f"weak_units: {clustering_stats['weak_units']}")
    print(f"clusters: {len(speaker_ids)}")
    print(f"wrote_rttm: {args.output_rttm}")
    print(f"wrote_output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
