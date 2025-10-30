#!/usr/bin/env bash
set -euo pipefail

# Repo root (directory of this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# 1) Ensure uv
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Ensure Python 3.12 and venv
uv python install 3.12 >/dev/null
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate

# 3) Choose PyTorch index URL
if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# 4) Install PyTorch (via uv, only if not already installed)
uv pip install --python ".venv/bin/python" --index-url "$TORCH_INDEX" torch torchvision torchaudio

# 5) Install misc libraries
uv pip install --python ".venv/bin/python" matplotlib torchlens

# 6) Verify
python - <<'PY'
import torch
print({
  'torch': torch.__version__,
  'cuda_available': torch.cuda.is_available(),
  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
})
PY

echo
echo "Environment ready. Activate with: source .venv/bin/activate"

