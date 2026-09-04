#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if ! command -v module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -f "${module_init}" ]]; then
      # shellcheck disable=SC1090
      source "${module_init}"
      break
    fi
  done
fi

if ! command -v module >/dev/null 2>&1; then
  echo "Environment Modules is unavailable; load gcc, Python, and ffmpeg manually before running this script." >&2
  exit 1
fi

module load gcc/13.3.0
module load python/3.13.1
module load apps/binapps/ffmpeg/4.1.3

if [[ ! -x "${PYTHON}" ]]; then
  echo "Project virtualenv not found: ${PYTHON}. Run setup.sh first." >&2
  exit 1
fi

SUBTITLE_DIR="${1:-/mnt/eps01-rds/Riza-Batista-Group/y39410ys/BBC-AVS-Dataset-ERA/subtitles}"
VIDEO_DIR="${2:-/mnt/eps01-rds/Riza-Batista-Group/y39410ys/BBC-AVS-Dataset-ERA/data}"

SPEAKER_MODEL="nvidia/speakerverification_en_titanet_large"
CLUSTERING_METHOD="constrained-spectral"
MIN_NUM_SPEAKERS="20"
MAX_NUM_SPEAKERS="80"
SPECTRAL_NEIGHBORS="30"
SPECTRAL_SIGMA="0.15"
DEVICE="cuda"
PROGRESS_EVERY="10"

if [[ ! -d "${SUBTITLE_DIR}" ]]; then
  echo "Subtitle directory not found: ${SUBTITLE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VIDEO_DIR}" ]]; then
  echo "Video directory not found: ${VIDEO_DIR}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/.cache/huggingface" "${REPO_ROOT}/.cache/xdg" "${REPO_ROOT}/.cache/matplotlib" "${REPO_ROOT}/.cache/nemo"
mkdir -p "${REPO_ROOT}/BBC-AVS-ERA/pred" "${REPO_ROOT}/BBC-AVS-ERA/clusters" "${REPO_ROOT}/BBC-AVS-ERA/embeddings"

ok_count=0
skip_count=0
fail_count=0

shopt -s nullglob
subtitle_files=("${SUBTITLE_DIR}"/*_offset.vtt)
shopt -u nullglob

if [[ ${#subtitle_files[@]} -eq 0 ]]; then
  echo "No subtitle files matching *_offset.vtt in ${SUBTITLE_DIR}" >&2
  exit 1
fi

slugify() {
  local value="$1"
  value="${value// /_}"
  value="$(echo "${value}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//')"
  printf "%s" "${value}"
}

for subtitle_path in "${subtitle_files[@]}"; do
  subtitle_file="$(basename "${subtitle_path}")"
  movie_stem="${subtitle_file%_offset.vtt}"

  output_key="$(slugify "${movie_stem}")"
  output_rttm="${REPO_ROOT}/BBC-AVS-ERA/pred/${output_key}_titanet_tracklet_pred.rttm"
  output_dir="${REPO_ROOT}/BBC-AVS-ERA/clusters/${output_key}_titanet_tracklet"
  embedding_dir="${REPO_ROOT}/BBC-AVS-ERA/embeddings/${output_key}_titanet_tracklet"

  if [[ -f "${output_rttm}" ]]; then
    echo "[skip] Already processed: ${movie_stem}" >&2
    ((skip_count+=1))
    continue
  fi

  video_path="${VIDEO_DIR}/${movie_stem}.ts"

  if [[ ! -f "${video_path}" ]]; then
    echo "[skip] Missing video for subtitle: ${subtitle_file}" >&2
    ((skip_count+=1))
    continue
  fi

  echo "[run] ${movie_stem}"
  echo "      subtitle: ${subtitle_path}"
  echo "      video:    ${video_path}"

  if env \
    HF_HOME="${REPO_ROOT}/.cache/huggingface" \
    XDG_CACHE_HOME="${REPO_ROOT}/.cache/xdg" \
    MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib" \
    NEMO_CACHE_DIR="${REPO_ROOT}/.cache/nemo" \
    "${PYTHON}" "${REPO_ROOT}/scripts/titanet_tracklet_diarize.py" \
      --subtitle "${subtitle_path}" \
      --video "${video_path}" \
      --output-rttm "${output_rttm}" \
      --output-dir "${output_dir}" \
      --embedding-dir "${embedding_dir}" \
      --speaker-model "${SPEAKER_MODEL}" \
      --clustering-method "${CLUSTERING_METHOD}" \
      --min-num-speakers "${MIN_NUM_SPEAKERS}" \
      --max-num-speakers "${MAX_NUM_SPEAKERS}" \
      --spectral-neighbors "${SPECTRAL_NEIGHBORS}" \
      --spectral-sigma "${SPECTRAL_SIGMA}" \
      --device "${DEVICE}" \
      --progress-every "${PROGRESS_EVERY}"; then
    ((ok_count+=1))
  else
    echo "[fail] ${movie_stem}" >&2
    ((fail_count+=1))
  fi
done

echo
echo "Batch complete"
echo "  Success: ${ok_count}"
echo "  Skipped: ${skip_count}"
echo "  Failed:  ${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
  exit 2
fi

exit 0