#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

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
  echo "Environment Modules is unavailable; load ffmpeg manually before running setup.sh." >&2
  exit 1
fi

module load apps/binapps/ffmpeg/4.1.3

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install "cmake<4"
"${PYTHON}" -m pip install torch torchaudio
"${PYTHON}" -m pip install transformers accelerate omegaconf "nemo_toolkit[asr]"

"${PYTHON}" -c "from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel; from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel; from nemo.collections.asr.parts.utils.speaker_utils import perform_clustering; print('NeMo diarization imports OK')"

echo "Environment ready: ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
