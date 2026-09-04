#!/usr/bin/env python3
"""Assign VTT color cues to mapped subtitle rows and write a gold RTTM file."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_JSON = Path("data/labels/American_Fiction_2023.json")
DEFAULT_VTT = Path("data/labels/American_Fiction_2023.vtt")
DEFAULT_OUTPUT_JSON = Path("data/labels/American_Fiction_2023_with_colors.json")
DEFAULT_OUTPUT_RTTM = Path("data/labels/American_Fiction_2023_gold.rttm")
DEFAULT_SPEAKER_MAP = Path("data/labels/American_Fiction_2023_speaker_map.json")
DEFAULT_FILE_ID = "American_Fiction_2023"

TIMING_RE = re.compile(r"^\s*(\d\d:\d\d:\d\d(?:\.\d+)?)\s+-->\s+(\d\d:\d\d:\d\d(?:\.\d+)?)")
SPAN_RE = re.compile(r"<c(?:\.([A-Za-z0-9_-]+))?[^>]*>(.*?)</c>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class VttSpan:
    start: float
    end: float
    color: str
    text: str
    normalized_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign color_cue metadata to subtitle JSON rows from a VTT file and "
            "generate a gold-standard RTTM from the JSON speaker labels."
        )
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--vtt", type=Path, default=DEFAULT_VTT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-rttm", type=Path, default=DEFAULT_OUTPUT_RTTM)
    parser.add_argument("--speaker-map", type=Path, default=DEFAULT_SPEAKER_MAP)
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument(
        "--overwrite-input-json",
        action="store_true",
        help="Also write the enriched payload back to --json.",
    )
    return parser.parse_args()


def seconds_from_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def plain_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = plain_text(str(text))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def parse_vtt_spans(path: Path) -> list[VttSpan]:
    spans: list[VttSpan] = []
    for block in cue_blocks(path.read_text(encoding="utf-8-sig")):
        timing_index = next((index for index, line in enumerate(block) if "-->" in line), None)
        if timing_index is None:
            continue
        match = TIMING_RE.match(block[timing_index])
        if match is None:
            continue
        start = seconds_from_timestamp(match.group(1))
        end = seconds_from_timestamp(match.group(2))
        if end <= start:
            continue
        cue_text = "\n".join(block[timing_index + 1 :])
        color_matches = list(SPAN_RE.finditer(cue_text))
        if not color_matches:
            text = plain_text(cue_text)
            if text:
                spans.append(VttSpan(start, end, "unknown", text, normalize_text(text)))
            continue
        for color_match in color_matches:
            color = (color_match.group(1) or "unknown").strip().lower()
            text = plain_text(color_match.group(2))
            if text:
                spans.append(VttSpan(start, end, color, text, normalize_text(text)))
    return spans


def overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def text_match_score(row_text: str, span_text: str, overlap: float) -> tuple[float, str | None]:
    if not row_text or not span_text:
        return 0.0, None
    if row_text == span_text:
        return 1000.0 + overlap, "text_exact"
    if row_text in span_text:
        return 800.0 + len(row_text) / max(len(span_text), 1) + overlap, "text_row_in_span"
    if span_text in row_text:
        return 600.0 + len(span_text) / max(len(row_text), 1) + overlap, "text_span_in_row"
    row_tokens = set(row_text.split())
    span_tokens = set(span_text.split())
    if not row_tokens or not span_tokens:
        return 0.0, None
    jaccard = len(row_tokens & span_tokens) / len(row_tokens | span_tokens)
    if jaccard >= 0.65:
        return 300.0 + jaccard + overlap, "text_token_overlap"
    return 0.0, None


def choose_color_for_row(row: dict[str, Any], spans: list[VttSpan]) -> dict[str, Any]:
    start = seconds_from_timestamp(str(row["start_time"]))
    end = seconds_from_timestamp(str(row["end_time"]))
    row_text = normalize_text(row.get("vtt_subtitle") or row.get("moviesum_subtitle"))
    candidates = [
        (span, overlap_seconds(start, end, span.start, span.end))
        for span in spans
        if overlap_seconds(start, end, span.start, span.end) > 1e-6
    ]
    if not candidates:
        return {
            "color_cue": "unknown",
            "color_cue_confidence": 0.0,
            "color_cue_ambiguous": True,
            "color_cue_match_method": "no_time_overlap",
            "color_cue_candidates": {},
        }

    text_scores: defaultdict[str, float] = defaultdict(float)
    text_methods: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overlap_scores: defaultdict[str, float] = defaultdict(float)
    for span, overlap in candidates:
        overlap_scores[span.color] += overlap
        score, method = text_match_score(row_text, span.normalized_text, overlap)
        if score > 0:
            text_scores[span.color] += score
            if method is not None:
                text_methods[span.color][method] += 1

    if text_scores:
        scores = dict(text_scores)
        method_pool = text_methods
    else:
        scores = dict(overlap_scores)
        method_pool = defaultdict(Counter)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_color, top_score = ranked[0]
    total_score = sum(score for _color, score in ranked)
    confidence = top_score / total_score if total_score > 0 else 0.0
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    ambiguous = math_is_close(top_score, second_score) or confidence < 0.60
    if text_scores:
        match_method = method_pool[top_color].most_common(1)[0][0]
    else:
        match_method = "time_overlap"

    return {
        "color_cue": top_color,
        "color_cue_confidence": round(confidence, 6),
        "color_cue_ambiguous": ambiguous,
        "color_cue_match_method": match_method,
        "color_cue_candidates": {color: round(score, 6) for color, score in ranked},
    }


def row_midpoint(row: dict[str, Any]) -> float:
    start = seconds_from_timestamp(str(row["start_time"]))
    end = seconds_from_timestamp(str(row["end_time"]))
    return (start + end) / 2.0


def resolve_ambiguous_rows_with_speaker_context(
    rows: list[dict[str, Any]],
    max_distance_seconds: float = 300.0,
) -> list[dict[str, Any]]:
    resolutions: list[dict[str, Any]] = []
    midpoints = [row_midpoint(row) for row in rows]

    for index, row in enumerate(rows):
        if not row.get("color_cue_ambiguous"):
            continue
        speaker = row.get("speaker")
        candidates = set((row.get("color_cue_candidates") or {}).keys())
        if not speaker or not candidates:
            continue

        local_scores: defaultdict[str, float] = defaultdict(float)
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            if other.get("speaker") != speaker:
                continue
            color = other.get("color_cue")
            if color not in candidates or other.get("color_cue_ambiguous"):
                continue
            distance = abs(midpoints[index] - midpoints[other_index])
            if distance > max_distance_seconds:
                continue
            confidence = float(other.get("color_cue_confidence") or 0.0)
            local_scores[str(color)] += confidence / (1.0 + distance)

        if not local_scores:
            continue

        ranked = sorted(local_scores.items(), key=lambda item: (-item[1], item[0]))
        top_color, top_score = ranked[0]
        total_score = sum(score for _color, score in ranked)
        context_confidence = top_score / total_score if total_score > 0 else 0.0
        if context_confidence < 0.60:
            continue
        old_color = row.get("color_cue")
        row["color_cue_initial"] = old_color
        row["color_cue"] = top_color
        row["color_cue_confidence"] = round(max(float(row.get("color_cue_confidence") or 0.0), context_confidence), 6)
        row["color_cue_ambiguous"] = context_confidence < 0.60
        row["color_cue_match_method"] = "speaker_local_context"
        row["color_cue_context_candidates"] = {
            color: round(score, 6) for color, score in ranked
        }
        resolutions.append(
            {
                "row_index": index + 1,
                "speaker": speaker,
                "old_color": old_color,
                "new_color": top_color,
                "context_confidence": round(context_confidence, 6),
            }
        )

    return resolutions


def math_is_close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9


def enrich_json_with_colors(payload: dict[str, Any], spans: list[VttSpan]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = payload.get("mappings_compact")
    if not isinstance(rows, list):
        raise ValueError("Expected JSON payload to contain list key 'mappings_compact'")

    enriched = json.loads(json.dumps(payload, ensure_ascii=False))
    method_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    ambiguous_rows: list[int] = []
    unknown_rows: list[int] = []

    for index, row in enumerate(enriched["mappings_compact"], start=1):
        assignment = choose_color_for_row(row, spans)
        row.update(assignment)

    context_resolutions = resolve_ambiguous_rows_with_speaker_context(enriched["mappings_compact"])

    for index, row in enumerate(enriched["mappings_compact"], start=1):
        method_counts[row["color_cue_match_method"]] += 1
        color_counts[row["color_cue"]] += 1
        if row["color_cue_ambiguous"]:
            ambiguous_rows.append(index)
        if row["color_cue"] == "unknown":
            unknown_rows.append(index)

    summary = {
        "rows": len(enriched["mappings_compact"]),
        "colors": dict(sorted(color_counts.items())),
        "match_methods": dict(sorted(method_counts.items())),
        "speaker_context_resolutions": context_resolutions,
        "ambiguous_rows": ambiguous_rows,
        "unknown_rows": unknown_rows,
    }
    return enriched, summary


def sanitize_speaker_name(name: str | None, used: set[str]) -> str:
    if name is None:
        base = "unknown"
    else:
        normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
        base = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
        if not base:
            base = "unknown"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_speaker_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    used: set[str] = set()
    speaker_map: dict[str, str] = {}
    speakers = sorted({row.get("speaker") for row in rows if row.get("speaker")})
    for speaker in speakers:
        speaker_map[str(speaker)] = sanitize_speaker_name(str(speaker), used)
    return speaker_map


def rttm_line(file_id: str, start: float, duration: float, speaker_id: str) -> str:
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"


def write_gold_rttm(
    rows: list[dict[str, Any]],
    path: Path,
    speaker_map: dict[str, str],
    file_id: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    skipped_rows: list[int] = []
    for index, row in enumerate(rows, start=1):
        speaker = row.get("speaker")
        if not speaker:
            skipped_rows.append(index)
            continue
        start = seconds_from_timestamp(str(row["start_time"]))
        end = seconds_from_timestamp(str(row["end_time"]))
        if end <= start:
            skipped_rows.append(index)
            continue
        lines.append(rttm_line(file_id, start, end - start, speaker_map[str(speaker)]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "rttm_turns": len(lines),
        "skipped_rows": skipped_rows,
        "speakers": len(speaker_map),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    spans = parse_vtt_spans(args.vtt)
    enriched, color_summary = enrich_json_with_colors(payload, spans)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.overwrite_input_json:
        args.json.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = enriched["mappings_compact"]
    speaker_map = build_speaker_map(rows)
    args.speaker_map.parent.mkdir(parents=True, exist_ok=True)
    args.speaker_map.write_text(json.dumps(speaker_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rttm_summary = write_gold_rttm(rows, args.output_rttm, speaker_map, args.file_id)

    print(f"vtt_spans: {len(spans)}")
    print(f"rows_colored: {color_summary['rows']}")
    print(f"color_counts: {color_summary['colors']}")
    print(f"match_methods: {color_summary['match_methods']}")
    print(f"ambiguous_rows: {len(color_summary['ambiguous_rows'])}")
    print(f"unknown_rows: {len(color_summary['unknown_rows'])}")
    print(f"rttm_turns: {rttm_summary['rttm_turns']}")
    print(f"rttm_skipped_rows: {rttm_summary['skipped_rows']}")
    print(f"speaker_count: {rttm_summary['speakers']}")
    print(f"wrote_json: {args.output_json}")
    print(f"wrote_rttm: {args.output_rttm}")
    print(f"wrote_speaker_map: {args.speaker_map}")


if __name__ == "__main__":
    main()
