#!/usr/bin/env bash
# Run after every RunPod restart:  source setup.sh
# Re-establishes the environment that lives on the (wiped) container disk.
# Code, venv, and HF model cache live on /workspace and persist on their own.

set -e

# --- Environment variables (wiped with ~/.bashrc on stop) ---
export HF_HOME=/workspace/hf_cache
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HF_HOME"

# --- Git identity + credential caching (~/.gitconfig is wiped on stop) ---
# Replace these two values with your own once; they then re-apply every restart.
git config --global user.name  "Hannah Tao"
git config --global user.email "hannahjtao@gmail.com"
# Cache the GitHub token after first push so you don't paste it every time:
git config --global credential.helper store
echo "git identity: $(git config --global user.name) <$(git config --global user.email)>"

# --- Reactivate the project virtualenv (lives in the repo, persists) ---
if [ -d "/workspace/compression-unlearning/.venv" ]; then
    source /workspace/compression-unlearning/.venv/bin/activate
    echo "venv activated: $(which python)"
else
    echo "WARNING: venv not found at /workspace/compression-unlearning/.venv"
fi

# --- Report cache state so you can see models survived ---
echo "HF_HOME = $HF_HOME"
if [ -d "$HF_HOME/hub" ]; then
    echo "Cached models:"
    ls "$HF_HOME/hub" | grep '^models--' || echo "  (none yet)"
else
    echo "  (no cache dir yet — models will download here)"
fi

# --- Reinstall Claude Code if the binary is gone (container disk wiped) ---
if ! command -v claude >/dev/null 2>&1; then
    echo "Claude Code not found — installing..."
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Claude Code present: $(command -v claude)"
fi

# --- GPU sanity check ---
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || echo "WARNING: nvidia-smi failed"

echo ""
echo "Setup complete. Start the agent with:  claude"
