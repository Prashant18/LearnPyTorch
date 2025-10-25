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

# 3) Choose PyTorch index URL based on CUDA
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)
  if [ -n "${CUDA_VER:-}" ]; then
    CUDA_MAJOR=${CUDA_VER%%.*}
    CUDA_MINOR=${CUDA_VER#*.}
    if [ "$CUDA_MAJOR" = "12" ]; then
      if   [ "$CUDA_MINOR" -ge 6 ]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126";
      elif [ "$CUDA_MINOR" -ge 4 ]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124";
      elif [ "$CUDA_MINOR" -ge 1 ]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"; fi
    fi
  fi
fi

# 4) Install PyTorch (via uv)
uv pip install --python ".venv/bin/python" --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio

# 4.1 Install misc libraries
uv pip install matplotlib


# 5) Verify
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

