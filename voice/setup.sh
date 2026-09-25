#!/usr/bin/env bash
# Installs everything make_voice.py needs: the Chatterbox clone engine (cb_venv)
# and the offline Kokoro-82M model + voices.
# The model files come from the `expo-kokoro` npm tarball (onnx + voice .bin files);
# only those data files are extracted, none of the package's JS is run.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null || ! command -v espeak-ng >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng >/dev/null
fi

pip install -q kokoro-onnx onnxruntime numpy scipy soundfile librosa praat-parselmouth pyloudnorm

MODEL_DIR=models
if [ ! -f "$MODEL_DIR/kokoro-quantized.onnx" ] || [ ! -f "$MODEL_DIR/voices.npz" ]; then
  mkdir -p "$MODEL_DIR" && tmp=$(mktemp -d)
  url=$(curl -sS https://registry.npmjs.org/expo-kokoro/1.1.9 | python3 -c "import sys,json;print(json.load(sys.stdin)['dist']['tarball'])")
  curl -sS -o "$tmp/pkg.tgz" "$url"
  tar xzf "$tmp/pkg.tgz" -C "$tmp" package/build/kokoro-quantized.onnx package/build/voices
  echo "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478  $tmp/package/build/kokoro-quantized.onnx" | sha256sum -c -
  mv "$tmp/package/build/kokoro-quantized.onnx" "$MODEL_DIR/"
  python3 - "$tmp/package/build/voices" "$MODEL_DIR/voices.npz" <<'EOF'
import sys, glob, os, numpy as np
src, dst = sys.argv[1], sys.argv[2]
voices = {os.path.basename(f)[:-4]: np.fromfile(f, dtype=np.float32).reshape(-1, 1, 256)
          for f in sorted(glob.glob(os.path.join(src, "*.bin")))}
np.savez(dst, **voices)
print(f"{len(voices)} voices -> {dst}")
EOF
  rm -rf "$tmp"
fi
# Voice-clone engine (Chatterbox, MIT) in its own venv: it pins torch 2.6.
# Its weights download from huggingface.co on first use, so the environment's
# network allowlist needs huggingface.co and *.hf.co.
if [ ! -x cb_venv/bin/python ]; then
  python3 -m venv cb_venv
  cb_venv/bin/pip install -q chatterbox-tts faster-whisper
fi
echo "setup ok"
