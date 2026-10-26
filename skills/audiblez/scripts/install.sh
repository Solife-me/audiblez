#!/usr/bin/env bash
# Installs audiblez (and ffmpeg) and writes its config. Safe to re-run: upgrades audiblez, keeps an existing config.
# usage: install.sh [TTS_BASE_URL] [TTS_MODEL] [VOICE] [LLM_MODEL]
set -euo pipefail

TTS_BASE_URL="${1:-http://MacBook-Pro-2.local:11434/v1}"
TTS_MODEL="${2:-phonon}"
VOICE="${3:-default}"
LLM_MODEL="${4:-gemma4:26b-mlx}"
REPO="${AUDIBLEZ_REPO:-git+https://github.com/Solife-me/audiblez}"

if ! command -v ffmpeg >/dev/null; then
  echo "Installing ffmpeg..."
  if command -v apt-get >/dev/null; then sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg espeak-ng
  elif command -v brew >/dev/null; then brew install ffmpeg espeak-ng
  elif command -v dnf >/dev/null; then sudo dnf install -y ffmpeg espeak-ng
  else echo "Please install ffmpeg with your package manager" >&2; exit 1; fi
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# audiblez needs Python 3.11/3.12 (Kokoro doesn't support 3.13 yet); uv fetches it if missing.
# CPU-only PyTorch: TTS runs on the Ollama server, so the multi-GB CUDA wheels are not needed here.
# faster-whisper lets audiblez listen to every clip and regenerate the rare ones where the TTS model drops or invents words.
echo "Installing audiblez from $REPO..."
uv tool install --force --python 3.12 --torch-backend cpu --with faster-whisper "$REPO"

CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/audiblez/config.toml"
if [ -f "$CONFIG" ]; then
  echo "Keeping existing config $CONFIG"
else
  mkdir -p "$(dirname "$CONFIG")"
  cat > "$CONFIG" <<EOF
[tts]
provider = "ollama"
model = "$TTS_MODEL"
voice = "$VOICE"
speed = 1.0

[llm]
provider = "ollama"
model = "$LLM_MODEL"
reasoning_effort = "none"

[providers.kokoro]
type = "kokoro"

[providers.ollama]
type = "openai"
base_url = "$TTS_BASE_URL"
EOF
  # Measured voice profiles (gender, age, accent, expressiveness) so characters can be cast well
  PROFILES="$(dirname "$0")/../references/$TTS_MODEL-voices.toml"
  if [ -f "$PROFILES" ]; then
    echo >> "$CONFIG"
    cat "$PROFILES" >> "$CONFIG"
    echo "Added the $TTS_MODEL voice profiles"
  fi
  echo "Wrote $CONFIG"
fi

echo "Checking the TTS server..."
if audiblez --list-voices | head -5; then
  echo "audiblez is ready."
else
  echo "audiblez is installed, but $TTS_BASE_URL did not answer. Check the URL (an IP address works when .local names don't resolve)." >&2
  exit 2
fi
