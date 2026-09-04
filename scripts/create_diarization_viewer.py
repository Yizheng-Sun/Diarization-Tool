#!/usr/bin/env python3
"""Generate an HTML viewer for subtitle-level diarization predictions."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_JSON = Path("data/labels/American_Fiction_2023_with_colors.json")
DEFAULT_RTTM = Path("data/labels/American_Fiction_2023_titanet_tracklet_pred.rttm")
DEFAULT_VIDEO = Path("data/movies/American_Fiction_2023.mp4")
DEFAULT_OUTPUT = Path("data/viewers/American_Fiction_2023_titanet_tracklet_viewer.html")
DEFAULT_TITLE = "American Fiction 2023 TitaNet Tracklet Diarization"

TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SubtitleRow:
    row_index: int
    start: float
    end: float
    text: str
    color: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class RttmTurn:
    file_id: str
    start: float
    end: float
    speaker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML movie viewer with predicted speaker labels over subtitles."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--rttm", type=Path, default=DEFAULT_RTTM)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--file-id", help="Optional RTTM file id filter.")
    return parser.parse_args()


def seconds_from_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def plain_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_subtitle_rows(path: Path) -> list[SubtitleRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("mappings_compact")
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected {path} to contain list key 'mappings_compact'")

    rows: list[SubtitleRow] = []
    for row_index, row in enumerate(raw_rows, start=1):
        start = seconds_from_timestamp(str(row["start_time"]))
        end = seconds_from_timestamp(str(row["end_time"]))
        if end <= start:
            continue
        text = row.get("vtt_subtitle") or row.get("moviesum_subtitle") or ""
        rows.append(
            SubtitleRow(
                row_index=row_index,
                start=start,
                end=end,
                text=plain_text(str(text)),
                color=str(row.get("color_cue") or "unknown"),
            )
        )
    return rows


def parse_rttm(path: Path, file_id: str | None = None) -> list[RttmTurn]:
    turns: list[RttmTurn] = []
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
        if duration <= 0:
            continue
        turns.append(RttmTurn(current_file_id, start, start + duration, parts[7]))
    return turns


def overlap_duration(row: SubtitleRow, turn: RttmTurn) -> float:
    return max(0.0, min(row.end, turn.end) - max(row.start, turn.start))


def best_speaker_for_row(row: SubtitleRow, turns: list[RttmTurn]) -> tuple[str, float]:
    best_speaker = "unknown"
    best_overlap = 0.0
    for turn in turns:
        if turn.end <= row.start:
            continue
        if turn.start >= row.end:
            break
        overlap = overlap_duration(row, turn)
        if overlap > best_overlap:
            best_speaker = turn.speaker
            best_overlap = overlap
    return best_speaker, best_overlap


def align_rows(rows: list[SubtitleRow], turns: list[RttmTurn]) -> list[dict[str, Any]]:
    sorted_turns = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
    aligned = []
    for row in rows:
        speaker, overlap = best_speaker_for_row(row, sorted_turns)
        aligned.append(
            {
                "rowIndex": row.row_index,
                "start": round(row.start, 6),
                "end": round(row.end, 6),
                "duration": round(row.duration, 6),
                "startLabel": format_time(row.start),
                "endLabel": format_time(row.end),
                "text": row.text,
                "speaker": speaker,
                "color": row.color,
                "speakerOverlap": round(overlap, 6),
            }
        )
    return aligned


def relative_url(path: Path, from_dir: Path) -> str:
    relative = os.path.relpath(path.resolve(), from_dir.resolve())
    return quote(relative.replace(os.sep, "/"), safe="/.:_-()")


def json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html(title: str, video_url: str, rows: list[dict[str, Any]], source_info: dict[str, str]) -> str:
    speakers = sorted({row["speaker"] for row in rows})
    payload = {
        "title": title,
        "videoUrl": video_url,
        "rows": rows,
        "speakers": speakers,
        "sourceInfo": source_info,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111315;
      --panel: #181b1f;
      --panel-strong: #20242a;
      --line: #343a42;
      --text: #f2f4f5;
      --muted: #a8b0b8;
      --focus: #78d0ff;
      --danger: #ffb26e;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
    }}
    .app {{
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #15181c;
    }}
    h1 {{
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: var(--panel);
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(360px, 28vw);
      min-height: 0;
    }}
    .stage {{
      position: relative;
      min-width: 0;
      background: #060708;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    video {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }}
    .subtitle-overlay {{
      position: absolute;
      left: 50%;
      bottom: 78px;
      transform: translateX(-50%);
      width: min(86%, 980px);
      display: none;
      text-align: center;
      pointer-events: none;
      text-shadow: 0 2px 12px rgba(0, 0, 0, 0.95), 0 1px 2px rgba(0, 0, 0, 0.95);
    }}
    .subtitle-overlay.active {{
      display: block;
    }}
    .overlay-speaker {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      max-width: 100%;
      margin-bottom: 8px;
      padding: 5px 10px;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.62);
      border: 1px solid rgba(255, 255, 255, 0.2);
      font-size: clamp(13px, 1.6vw, 18px);
      font-weight: 800;
      color: white;
    }}
    .overlay-dot {{
      width: 10px;
      height: 10px;
      flex: 0 0 auto;
      border-radius: 50%;
      background: hsl(var(--speaker-hue), 78%, 58%);
      box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.16);
    }}
    .overlay-text {{
      margin: 0;
      font-size: clamp(23px, 3.1vw, 44px);
      font-weight: 800;
      line-height: 1.16;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }}
    .transcript {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border-left: 1px solid var(--line);
      background: var(--panel);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 132px auto auto;
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      background: #15181c;
    }}
    input, select, button {{
      min-width: 0;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-strong);
      color: var(--text);
      font: inherit;
      font-size: 13px;
    }}
    input, select {{
      padding: 0 10px;
    }}
    button {{
      width: 36px;
      cursor: pointer;
      font-size: 20px;
      line-height: 1;
    }}
    button:hover, input:focus, select:focus {{
      outline: 1px solid var(--focus);
      border-color: var(--focus);
    }}
    .rows {{
      overflow: auto;
      min-height: 0;
    }}
    .row {{
      display: grid;
      grid-template-columns: 86px 1fr;
      gap: 10px;
      width: 100%;
      padding: 10px;
      border: 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .row:hover {{
      background: rgba(255, 255, 255, 0.045);
      outline: none;
    }}
    .row.active {{
      background: rgba(120, 208, 255, 0.12);
      box-shadow: inset 3px 0 0 hsl(var(--speaker-hue), 78%, 58%);
    }}
    .row.filtered {{
      display: none;
    }}
    .time {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }}
    .line {{
      min-width: 0;
      display: grid;
      gap: 5px;
    }}
    .speaker {{
      width: fit-content;
      max-width: 100%;
      border: 1px solid color-mix(in srgb, hsl(var(--speaker-hue), 78%, 58%) 70%, white 10%);
      border-radius: 5px;
      padding: 3px 7px;
      background: color-mix(in srgb, hsl(var(--speaker-hue), 78%, 58%) 20%, black 80%);
      color: white;
      font-size: 12px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }}
    .text {{
      font-size: 14px;
      line-height: 1.38;
      overflow-wrap: anywhere;
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 11px;
    }}
    .empty {{
      display: none;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
    }}
    .empty.active {{
      display: block;
    }}
    @media (max-width: 900px) {{
      .app {{
        height: auto;
        min-height: 100vh;
        overflow: visible;
      }}
      main {{
        grid-template-columns: 1fr;
        grid-template-rows: 58vh minmax(360px, 42vh);
      }}
      .transcript {{
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
      .toolbar {{
        grid-template-columns: 1fr 112px auto auto;
      }}
      header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .stats {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>{html.escape(title)}</h1>
      <div class="stats" id="stats"></div>
    </header>
    <main>
      <section class="stage">
        <video id="video" controls preload="metadata" src="{html.escape(video_url)}"></video>
        <div class="subtitle-overlay" id="overlay">
          <div class="overlay-speaker" id="overlaySpeaker"><span class="overlay-dot"></span><span></span></div>
          <p class="overlay-text" id="overlayText"></p>
        </div>
      </section>
      <aside class="transcript">
        <div class="toolbar">
          <input id="search" type="search" placeholder="Search">
          <select id="speakerFilter"></select>
          <button id="previous" type="button" title="Previous subtitle" aria-label="Previous subtitle">‹</button>
          <button id="next" type="button" title="Next subtitle" aria-label="Next subtitle">›</button>
        </div>
        <div class="rows" id="rows"></div>
        <div class="empty" id="empty">No matching subtitles</div>
      </aside>
    </main>
  </div>
  <script id="viewer-data" type="application/json">{json_script_payload(payload)}</script>
  <script>
    const data = JSON.parse(document.getElementById("viewer-data").textContent);
    const rows = data.rows;
    const speakers = data.speakers;
    const video = document.getElementById("video");
    const rowsEl = document.getElementById("rows");
    const overlay = document.getElementById("overlay");
    const overlaySpeaker = document.getElementById("overlaySpeaker");
    const overlaySpeakerText = overlaySpeaker.querySelector("span:last-child");
    const overlayText = document.getElementById("overlayText");
    const search = document.getElementById("search");
    const speakerFilter = document.getElementById("speakerFilter");
    const previous = document.getElementById("previous");
    const next = document.getElementById("next");
    const empty = document.getElementById("empty");
    const stats = document.getElementById("stats");
    let activeIndex = -1;

    function speakerHue(label) {{
      let hash = 0;
      for (let i = 0; i < label.length; i += 1) {{
        hash = ((hash << 5) - hash + label.charCodeAt(i)) | 0;
      }}
      return Math.abs(hash) % 360;
    }}

    function setSpeakerHue(element, speaker) {{
      element.style.setProperty("--speaker-hue", String(speakerHue(speaker)));
    }}

    function shortTime(seconds) {{
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);
      return `${{String(hours).padStart(2, "0")}}:${{String(minutes).padStart(2, "0")}}:${{String(secs).padStart(2, "0")}}`;
    }}

    function rowMatches(row) {{
      const query = search.value.trim().toLowerCase();
      const selectedSpeaker = speakerFilter.value;
      const queryMatch = !query || row.text.toLowerCase().includes(query) || row.speaker.toLowerCase().includes(query);
      const speakerMatch = !selectedSpeaker || row.speaker === selectedSpeaker;
      return queryMatch && speakerMatch;
    }}

    function renderStats() {{
      const labeledRows = rows.filter((row) => row.speaker !== "unknown").length;
      const totalDuration = rows.reduce((sum, row) => sum + row.duration, 0);
      stats.innerHTML = "";
      [
        `${{rows.length}} subtitles`,
        `${{speakers.length}} speakers`,
        `${{labeledRows}} labeled`,
        `${{shortTime(totalDuration)}} subtitle time`
      ].forEach((text) => {{
        const span = document.createElement("span");
        span.className = "stat";
        span.textContent = text;
        stats.appendChild(span);
      }});
    }}

    function renderSpeakerFilter() {{
      speakerFilter.innerHTML = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = "All speakers";
      speakerFilter.appendChild(all);
      speakers.forEach((speaker) => {{
        const option = document.createElement("option");
        option.value = speaker;
        option.textContent = speaker;
        speakerFilter.appendChild(option);
      }});
    }}

    function renderRows() {{
      rowsEl.innerHTML = "";
      rows.forEach((row, index) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "row";
        button.dataset.index = String(index);
        setSpeakerHue(button, row.speaker);
        button.innerHTML = `
          <div class="time">${{shortTime(row.start)}}<br>${{shortTime(row.end)}}</div>
          <div class="line">
            <div class="speaker"></div>
            <div class="text"></div>
            <div class="meta"><span>row ${{row.rowIndex}}</span><span>color ${{row.color}}</span></div>
          </div>
        `;
        button.querySelector(".speaker").textContent = row.speaker;
        button.querySelector(".text").textContent = row.text || "(empty subtitle)";
        button.addEventListener("click", () => {{
          video.currentTime = Math.max(0, row.start + 0.01);
          video.play().catch(() => {{}});
          setActiveIndex(index, true);
        }});
        rowsEl.appendChild(button);
      }});
      applyFilters();
    }}

    function applyFilters() {{
      let visible = 0;
      rows.forEach((row, index) => {{
        const element = rowsEl.children[index];
        const matches = rowMatches(row);
        element.classList.toggle("filtered", !matches);
        if (matches) visible += 1;
      }});
      empty.classList.toggle("active", visible === 0);
    }}

    function findActiveIndex(time) {{
      let low = 0;
      let high = rows.length - 1;
      let candidate = -1;
      while (low <= high) {{
        const mid = Math.floor((low + high) / 2);
        if (rows[mid].start <= time) {{
          candidate = mid;
          low = mid + 1;
        }} else {{
          high = mid - 1;
        }}
      }}
      if (candidate >= 0 && time < rows[candidate].end) {{
        return candidate;
      }}
      return -1;
    }}

    function setActiveIndex(index, forceScroll = false) {{
      if (activeIndex === index && !forceScroll) return;
      if (activeIndex >= 0 && rowsEl.children[activeIndex]) {{
        rowsEl.children[activeIndex].classList.remove("active");
      }}
      activeIndex = index;
      if (activeIndex < 0) {{
        overlay.classList.remove("active");
        return;
      }}
      const row = rows[activeIndex];
      const rowElement = rowsEl.children[activeIndex];
      setSpeakerHue(overlaySpeaker, row.speaker);
      overlaySpeakerText.textContent = row.speaker;
      overlayText.textContent = row.text;
      overlay.classList.add("active");
      rowElement.classList.add("active");
      if (forceScroll || !rowElement.classList.contains("filtered")) {{
        rowElement.scrollIntoView({{ block: "nearest" }});
      }}
    }}

    function seekToNeighbor(direction) {{
      const visible = rows
        .map((row, index) => [row, index])
        .filter(([row]) => rowMatches(row))
        .map(([, index]) => index);
      if (!visible.length) return;
      const currentTime = video.currentTime;
      let target = direction > 0 ? visible[0] : visible[visible.length - 1];
      if (direction > 0) {{
        target = visible.find((index) => rows[index].start > currentTime + 0.05) ?? visible[0];
      }} else {{
        const previousRows = visible.filter((index) => rows[index].start < currentTime - 0.05);
        target = previousRows.length ? previousRows[previousRows.length - 1] : visible[visible.length - 1];
      }}
      video.currentTime = Math.max(0, rows[target].start + 0.01);
      setActiveIndex(target, true);
    }}

    video.addEventListener("timeupdate", () => setActiveIndex(findActiveIndex(video.currentTime)));
    video.addEventListener("seeked", () => setActiveIndex(findActiveIndex(video.currentTime), true));
    search.addEventListener("input", applyFilters);
    speakerFilter.addEventListener("change", applyFilters);
    previous.addEventListener("click", () => seekToNeighbor(-1));
    next.addEventListener("click", () => seekToNeighbor(1));

    renderStats();
    renderSpeakerFilter();
    renderRows();
  </script>
</body>
</html>
"""


def write_viewer(
    json_path: Path,
    rttm_path: Path,
    video_path: Path,
    output_path: Path,
    title: str,
    file_id: str | None = None,
) -> dict[str, Any]:
    rows = parse_subtitle_rows(json_path)
    turns = parse_rttm(rttm_path, file_id=file_id)
    if not rows:
        raise ValueError(f"No subtitle rows found in {json_path}")
    if not turns:
        raise ValueError(f"No RTTM turns found in {rttm_path}")

    aligned_rows = align_rows(rows, turns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_url = relative_url(video_path, output_path.parent)
    source_info = {
        "json": str(json_path),
        "rttm": str(rttm_path),
        "video": str(video_path),
        "file_id": file_id or "",
    }
    html_text = render_html(title, video_url, aligned_rows, source_info)
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "output": str(output_path),
        "rows": len(aligned_rows),
        "speakers": len({row["speaker"] for row in aligned_rows}),
        "unlabeled_rows": sum(1 for row in aligned_rows if row["speaker"] == "unknown"),
    }


def main() -> None:
    args = parse_args()
    summary = write_viewer(
        args.json,
        args.rttm,
        args.video,
        args.output,
        args.title,
        file_id=args.file_id,
    )
    print(f"wrote_viewer: {summary['output']}")
    print(f"rows: {summary['rows']}")
    print(f"speakers: {summary['speakers']}")
    print(f"unlabeled_rows: {summary['unlabeled_rows']}")


if __name__ == "__main__":
    main()
