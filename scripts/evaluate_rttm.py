#!/usr/bin/env python3
"""Evaluate a predicted RTTM file against a gold/reference RTTM file.

This is a dependency-light RTTM evaluator. It computes an optimal one-to-one
mapping from hypothesis speakers to reference speakers by maximizing overlap
time, then reports DER-style miss, false alarm, and confusion rates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPSILON = 1e-9
UNKNOWN_SPEAKERS = {"", "<NA>", "unknown", "unk", "none", "null"}


@dataclass(frozen=True)
class Turn:
    file_id: str
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a predicted RTTM against a gold RTTM using an optimal "
            "speaker-label mapping."
        )
    )
    parser.add_argument("--gold", "--reference", dest="gold", type=Path, required=True)
    parser.add_argument("--predicted", "--hypothesis", dest="predicted", type=Path, required=True)
    parser.add_argument("--file-id", help="Evaluate only one RTTM file id.")
    parser.add_argument(
        "--ignore-speaker",
        action="append",
        default=[],
        help="Speaker label to ignore. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Ignore unknown-style labels such as unknown, unk, none, null, and <NA>.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to write metrics JSON.")
    parser.add_argument("--show-mapping", action="store_true", help="Print hypothesis-to-gold speaker mapping.")
    return parser.parse_args()


def parse_rttm(
    path: Path,
    file_id: str | None = None,
    ignore_speakers: set[str] | None = None,
) -> list[Turn]:
    ignore_speakers = ignore_speakers or set()
    turns: list[Turn] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM line {path}:{line_number}: {raw_line}")
        current_file_id = parts[1]
        if file_id is not None and current_file_id != file_id:
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        if duration <= 0:
            continue
        if speaker in ignore_speakers or speaker.lower() in ignore_speakers:
            continue
        turns.append(Turn(current_file_id, start, start + duration, speaker))
    return turns


def active_speakers(turns: list[Turn], start: float, end: float) -> set[str]:
    active = set()
    for turn in turns:
        if min(end, turn.end) - max(start, turn.start) > EPSILON:
            active.add(turn.speaker)
    return active


def boundaries(reference: list[Turn], hypothesis: list[Turn]) -> list[float]:
    values = {turn.start for turn in reference}
    values.update(turn.end for turn in reference)
    values.update(turn.start for turn in hypothesis)
    values.update(turn.end for turn in hypothesis)
    return sorted(values)


def maximize_assignment(weights: list[list[float]]) -> list[int]:
    """Return a max-weight assignment for rows to columns, assuming rows <= columns."""
    if not weights:
        return []
    row_count = len(weights)
    col_count = len(weights[0])
    if row_count > col_count:
        raise ValueError("maximize_assignment requires row_count <= col_count")

    max_weight = max((weight for row in weights for weight in row), default=0.0)
    costs = [[max_weight - weight for weight in row] for row in weights]
    potentials_row = [0.0] * (row_count + 1)
    potentials_col = [0.0] * (col_count + 1)
    matching = [0] * (col_count + 1)
    way = [0] * (col_count + 1)

    for row in range(1, row_count + 1):
        matching[0] = row
        col0 = 0
        min_values = [math.inf] * (col_count + 1)
        used = [False] * (col_count + 1)
        while True:
            used[col0] = True
            row0 = matching[col0]
            delta = math.inf
            col1 = 0
            for col in range(1, col_count + 1):
                if used[col]:
                    continue
                current = costs[row0 - 1][col - 1] - potentials_row[row0] - potentials_col[col]
                if current < min_values[col]:
                    min_values[col] = current
                    way[col] = col0
                if min_values[col] < delta:
                    delta = min_values[col]
                    col1 = col
            for col in range(0, col_count + 1):
                if used[col]:
                    potentials_row[matching[col]] += delta
                    potentials_col[col] -= delta
                else:
                    min_values[col] -= delta
            col0 = col1
            if matching[col0] == 0:
                break
        while True:
            col1 = way[col0]
            matching[col0] = matching[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * row_count
    for col in range(1, col_count + 1):
        if matching[col] != 0:
            assignment[matching[col] - 1] = col - 1
    return assignment


def optimal_speaker_mapping(
    reference_speakers: list[str],
    hypothesis_speakers: list[str],
    overlap: dict[tuple[str, str], float],
) -> dict[str, str]:
    if not reference_speakers or not hypothesis_speakers:
        return {}

    if len(hypothesis_speakers) <= len(reference_speakers):
        weights = [
            [overlap.get((ref, hyp), 0.0) for ref in reference_speakers]
            for hyp in hypothesis_speakers
        ]
        assignment = maximize_assignment(weights)
        return {
            hypothesis_speakers[row]: reference_speakers[col]
            for row, col in enumerate(assignment)
            if col >= 0
        }

    weights = [
        [overlap.get((ref, hyp), 0.0) for hyp in hypothesis_speakers]
        for ref in reference_speakers
    ]
    assignment = maximize_assignment(weights)
    return {
        hypothesis_speakers[col]: reference_speakers[row]
        for row, col in enumerate(assignment)
        if col >= 0
    }


def evaluate(reference: list[Turn], hypothesis: list[Turn]) -> dict[str, Any]:
    timeline = boundaries(reference, hypothesis)
    if len(timeline) < 2:
        raise ValueError("No evaluable RTTM time span found")

    reference_speakers = sorted({turn.speaker for turn in reference})
    hypothesis_speakers = sorted({turn.speaker for turn in hypothesis})
    overlap: defaultdict[tuple[str, str], float] = defaultdict(float)

    reference_speaker_time = 0.0
    hypothesis_speaker_time = 0.0
    evaluated_time = 0.0
    for start, end in zip(timeline, timeline[1:]):
        duration = end - start
        if duration <= EPSILON:
            continue
        ref_active = active_speakers(reference, start, end)
        hyp_active = active_speakers(hypothesis, start, end)
        if not ref_active and not hyp_active:
            continue
        evaluated_time += duration
        reference_speaker_time += duration * len(ref_active)
        hypothesis_speaker_time += duration * len(hyp_active)
        for ref_speaker in ref_active:
            for hyp_speaker in hyp_active:
                overlap[(ref_speaker, hyp_speaker)] += duration

    if reference_speaker_time <= EPSILON:
        raise ValueError("Reference RTTM has no speaker time after filtering")

    mapping = optimal_speaker_mapping(reference_speakers, hypothesis_speakers, overlap)

    correct = 0.0
    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    for start, end in zip(timeline, timeline[1:]):
        duration = end - start
        if duration <= EPSILON:
            continue
        ref_active = active_speakers(reference, start, end)
        hyp_active = active_speakers(hypothesis, start, end)
        if not ref_active and not hyp_active:
            continue
        correct_count = sum(1 for hyp_speaker in hyp_active if mapping.get(hyp_speaker) in ref_active)
        missed_count = max(0, len(ref_active) - len(hyp_active))
        false_alarm_count = max(0, len(hyp_active) - len(ref_active))
        confusion_count = max(0, min(len(ref_active), len(hyp_active)) - correct_count)
        correct += duration * correct_count
        missed += duration * missed_count
        false_alarm += duration * false_alarm_count
        confusion += duration * confusion_count

    der_ratio = (missed + false_alarm + confusion) / reference_speaker_time
    speaker_label_recall = correct / reference_speaker_time
    speaker_label_precision = (
        correct / hypothesis_speaker_time if hypothesis_speaker_time > EPSILON else 0.0
    )
    if speaker_label_precision + speaker_label_recall > EPSILON:
        speaker_label_f1 = (
            2.0
            * speaker_label_precision
            * speaker_label_recall
            / (speaker_label_precision + speaker_label_recall)
        )
    else:
        speaker_label_f1 = 0.0
    return {
        "reference_turns": len(reference),
        "hypothesis_turns": len(hypothesis),
        "reference_speakers": len(reference_speakers),
        "hypothesis_speakers": len(hypothesis_speakers),
        "mapped_speakers": len(mapping),
        "evaluated_time_sec": round(evaluated_time, 6),
        "reference_speaker_time_sec": round(reference_speaker_time, 6),
        "hypothesis_speaker_time_sec": round(hypothesis_speaker_time, 6),
        "correct_speaker_time_sec": round(correct, 6),
        "missed_speaker_time_sec": round(missed, 6),
        "false_alarm_speaker_time_sec": round(false_alarm, 6),
        "confusion_speaker_time_sec": round(confusion, 6),
        "der_ratio": round(der_ratio, 8),
        "der": round(der_ratio, 8),
        "der_percent": round(der_ratio * 100.0, 4),
        "speaker_label_precision": round(speaker_label_precision, 8),
        "speaker_label_precision_percent": round(speaker_label_precision * 100.0, 4),
        "speaker_label_recall": round(speaker_label_recall, 8),
        "speaker_label_recall_percent": round(speaker_label_recall * 100.0, 4),
        "speaker_label_f1": round(speaker_label_f1, 8),
        "speaker_label_f1_percent": round(speaker_label_f1 * 100.0, 4),
        "speaker_time_accuracy": round(speaker_label_recall, 8),
        "speaker_time_accuracy_percent": round(speaker_label_recall * 100.0, 4),
        "mapping": dict(sorted(mapping.items())),
    }


def print_metrics(metrics: dict[str, Any], show_mapping: bool) -> None:
    ordered_keys = [
        "reference_turns",
        "hypothesis_turns",
        "reference_speakers",
        "hypothesis_speakers",
        "mapped_speakers",
        "evaluated_time_sec",
        "reference_speaker_time_sec",
        "hypothesis_speaker_time_sec",
        "correct_speaker_time_sec",
        "missed_speaker_time_sec",
        "false_alarm_speaker_time_sec",
        "confusion_speaker_time_sec",
        "der_ratio",
        "der_percent",
        "speaker_label_precision_percent",
        "speaker_label_recall_percent",
        "speaker_label_f1_percent",
        "speaker_time_accuracy_percent",
    ]
    for key in ordered_keys:
        print(f"{key}: {metrics[key]}")
    if show_mapping:
        print("speaker_mapping:")
        for hyp_speaker, ref_speaker in metrics["mapping"].items():
            print(f"  {hyp_speaker} -> {ref_speaker}")


def main() -> None:
    args = parse_args()
    ignore_speakers = set(args.ignore_speaker)
    if args.ignore_unknown:
        ignore_speakers.update(UNKNOWN_SPEAKERS)
    reference = parse_rttm(args.gold, file_id=args.file_id, ignore_speakers=ignore_speakers)
    hypothesis = parse_rttm(args.predicted, file_id=args.file_id, ignore_speakers=ignore_speakers)
    metrics = evaluate(reference, hypothesis)
    print_metrics(metrics, show_mapping=args.show_mapping)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
