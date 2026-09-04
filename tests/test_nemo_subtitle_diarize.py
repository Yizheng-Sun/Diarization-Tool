from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nemo_subtitle_diarize.py"
SPEC = importlib.util.spec_from_file_location("nemo_subtitle_diarize", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
nemo_subtitle_diarize = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nemo_subtitle_diarize
SPEC.loader.exec_module(nemo_subtitle_diarize)


class NemoSubtitleDiarizeTests(unittest.TestCase):
    def test_progress_logger_prints_elapsed_prefix(self) -> None:
        stream = io.StringIO()
        progress = nemo_subtitle_diarize.ProgressLogger(stream=stream)
        progress.log("hello")

        self.assertRegex(stream.getvalue(), r"^\[progress \d\d:\d\d:\d\d\] hello\n$")

    def test_require_program_uses_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_bin = Path(tmpdir) / "ffmpeg"
            ffmpeg_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(ffmpeg_bin, 0o755)

            resolved = nemo_subtitle_diarize.require_program("ffmpeg", explicit_path=ffmpeg_bin)

        self.assertEqual(resolved, str(ffmpeg_bin))

    def test_require_program_uses_directory_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_bin = Path(tmpdir) / "ffmpeg"
            ffmpeg_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(ffmpeg_bin, 0o755)

            resolved = nemo_subtitle_diarize.require_program("ffmpeg", explicit_path=Path(tmpdir))

        self.assertEqual(resolved, str(ffmpeg_bin))

    def test_require_program_reports_module_hint_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, r"module system.*--ffmpeg-bin"):
                nemo_subtitle_diarize.require_program("ffmpeg")

    def test_seconds_from_timestamp(self) -> None:
        self.assertAlmostEqual(
            nemo_subtitle_diarize.seconds_from_timestamp("01:02:03.500"),
            3723.5,
        )

    def test_parse_subtitle_rows_strips_tags_and_limits(self) -> None:
        payload = {
            "mappings_compact": [
                {
                    "start_time": "00:00:01.000",
                    "end_time": "00:00:02.250",
                    "vtt_subtitle": "<i>Hello</i> &amp; goodbye",
                    "speaker": "Gold Name",
                    "color_cue": "white",
                    "color_cue_confidence": 0.9,
                    "color_cue_ambiguous": False,
                },
                {
                    "start_time": "00:00:03.000",
                    "end_time": "00:00:04.000",
                    "vtt_subtitle": "Second",
                    "speaker": "Other",
                    "color_cue": "yellow",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = nemo_subtitle_diarize.parse_subtitle_rows(path, max_rows=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].row_index, 1)
        self.assertEqual(rows[0].text, "Hello & goodbye")
        self.assertEqual(rows[0].gold_speaker, "Gold Name")

    def test_build_scale_windows_keeps_exact_row_timestamps_separate(self) -> None:
        row = nemo_subtitle_diarize.SubtitleRow(
            row_index=1,
            start=10.0,
            end=11.0,
            color="white",
            text="hello",
        )
        windows = nemo_subtitle_diarize.build_scale_windows(
            [row],
            [3.0, 0.5],
            audio_duration=20.0,
            min_window=0.5,
        )

        self.assertEqual(windows[0][0].start, 9.0)
        self.assertEqual(windows[0][0].end, 12.0)
        self.assertEqual(windows[1][0].start, 10.0)
        self.assertEqual(windows[1][0].end, 11.0)
        self.assertEqual(row.start, 10.0)
        self.assertEqual(row.end, 11.0)

    def test_build_scale_windows_clips_to_audio_bounds(self) -> None:
        row = nemo_subtitle_diarize.SubtitleRow(
            row_index=1,
            start=0.1,
            end=0.3,
            color="white",
            text="hello",
        )
        windows = nemo_subtitle_diarize.build_scale_windows(
            [row],
            [3.0],
            audio_duration=2.0,
            min_window=0.5,
        )

        self.assertEqual(windows[0][0].start, 0.0)
        self.assertEqual(windows[0][0].end, 2.0)

    def test_rttm_line_format(self) -> None:
        self.assertEqual(
            nemo_subtitle_diarize.rttm_line("movie", 1.2345, 2.3456, "speaker_001"),
            "SPEAKER movie 1 1.234 2.346 <NA> <NA> speaker_001 <NA> <NA>",
        )

    def test_adjacent_same_color_rows_are_forced_to_one_cluster(self) -> None:
        rows = [
            nemo_subtitle_diarize.SubtitleRow(
                row_index=1,
                start=0.0,
                end=1.0,
                color="white",
                text="a",
            ),
            nemo_subtitle_diarize.SubtitleRow(
                row_index=2,
                start=1.0,
                end=4.0,
                color="WHITE",
                text="b",
            ),
            nemo_subtitle_diarize.SubtitleRow(
                row_index=3,
                start=4.0,
                end=5.0,
                color="yellow",
                text="c",
            ),
            nemo_subtitle_diarize.SubtitleRow(
                row_index=4,
                start=5.0,
                end=6.0,
                color="unknown",
                text="d",
            ),
            nemo_subtitle_diarize.SubtitleRow(
                row_index=5,
                start=6.0,
                end=7.0,
                color="unknown",
                text="e",
            ),
        ]

        refined, audit_rows, stats = nemo_subtitle_diarize.apply_adjacent_color_constraints(
            rows,
            [10, 20, 30, 40, 50],
        )

        self.assertEqual(refined, [20, 20, 30, 40, 50])
        self.assertEqual(stats["constrained_runs"], 1)
        self.assertEqual(stats["changed_row_indices"], [1])
        self.assertEqual(audit_rows[0]["row_indices"], [1, 2])

    def test_write_outputs_keeps_one_rttm_turn_per_row(self) -> None:
        rows = [
            nemo_subtitle_diarize.SubtitleRow(
                row_index=1,
                start=1.0,
                end=2.0,
                color="white",
                text="hello",
            ),
            nemo_subtitle_diarize.SubtitleRow(
                row_index=2,
                start=3.0,
                end=4.5,
                color="yellow",
                text="there",
            ),
        ]
        labels = [7, 7]
        speaker_ids = {7: 1}
        details = [
            {"best_similarity": 0.5, "second_best_similarity": 0.1, "margin": 0.4},
            {"best_similarity": 0.4, "second_best_similarity": 0.2, "margin": 0.2},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            nemo_subtitle_diarize.write_outputs(
                output_dir=out,
                output_rttm=out / "final.rttm",
                rows=rows,
                labels=labels,
                speaker_ids=speaker_ids,
                similarity_details=details,
                refinement_stats={"enabled": False, "relabeled_rows": 0},
                color_refinement_stats={
                    "enabled": True,
                    "changed_rows": 0,
                    "changed_row_indices": [],
                },
                scale_windows=[3.0, 0.5],
                speaker_model="titanet_large",
                msdd_model="diar_msdd_telephonic",
                rttm_file_id="movie",
                clustering_config={"max_num_speakers": 80},
            )
            rttm_lines = (out / "speaker_segments.rttm").read_text(encoding="utf-8").splitlines()
            final_rttm_lines = (out / "final.rttm").read_text(encoding="utf-8").splitlines()
            jsonl_lines = (out / "speaker_lines.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rttm_lines), 2)
        self.assertEqual(final_rttm_lines, rttm_lines)
        self.assertEqual(len(jsonl_lines), 2)
        self.assertIn("speaker_001", rttm_lines[0])


if __name__ == "__main__":
    unittest.main()
