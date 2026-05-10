#!/usr/bin/env bash
# A10 bootstrap — installs everything needed for icefall zipformer fine-tuning.
# Idempotent: safe to re-run if a step fails partway through.
#
# Usage on a fresh Lambda Labs A10 (Ubuntu 22.04, Lambda Stack):
#   bash ~/bootstrap_a10.sh
#
# Prereqs uploaded to ~ before running:
#   ~/voice_data_v2.tar.gz   (built locally — audio + lhotse cuts + fbank features)

set -euo pipefail

echo "=== [1/8] Verify GPU + base PyTorch ==="
nvidia-smi
python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"

echo
echo "=== [2/8] Pin numpy<2 (system TF was compiled against numpy 1.x) ==="
pip install --quiet "numpy<2"

echo
echo "=== [3/8] Upgrade PyTorch to 2.11.0 + CUDA 12.8 (matches available k2 wheel) ==="
pip install --quiet torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo
echo "=== [4/8] Install k2, lhotse, hf hub ==="
pip install --quiet \
  "k2==1.24.4.dev20260423+cuda12.8.torch2.11.0" \
  -f https://k2-fsa.github.io/k2/cuda.html
pip install --quiet lhotse "huggingface-hub>=1.0" tensorboard

echo
echo "=== [5/8] Clone icefall + install deps ==="
if [[ ! -d ~/icefall ]]; then
  git clone https://github.com/k2-fsa/icefall.git ~/icefall
fi
cd ~/icefall
pip install --quiet -r requirements.txt
if ! grep -q "icefall" ~/.bashrc; then
  echo "export PYTHONPATH=\$HOME/icefall:\$PYTHONPATH" >>~/.bashrc
fi
export PYTHONPATH=$HOME/icefall:${PYTHONPATH:-}

echo
echo "=== [6/8] Download base checkpoint (Candidate B — recipe-matched) ==="
if [[ ! -f ~/base_ckpt_B/exp/pretrained.pt ]]; then
  hf download \
    Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 \
    exp/pretrained.pt \
    --local-dir ~/base_ckpt_B
  hf download \
    Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 \
    --include "data/lang_bpe_500/*" \
    --local-dir ~/base_ckpt_B
else
  echo "  Already downloaded."
fi

echo
echo "=== [7/8] Extract data tarball + fix ownership ==="
if [[ ! -d /Users/skinnercheng/Projects/VOX_Personalis/out/lhotse_manifests ]]; then
  sudo tar xzf ~/voice_data_v2.tar.gz -C /
  sudo chown -R ubuntu:ubuntu /Users/
else
  echo "  Already extracted."
fi

echo
echo "=== [8/8] Patch icefall finetune.py + symlink data dir ==="
cd ~/icefall/egs/librispeech/ASR

# Patch num_frames None handling (idempotent — only patches if not already done)
python3 <<'PY'
from pathlib import Path
p = Path('./zipformer/finetune.py')
src = p.read_text()
old = 'T = ((c.num_frames - 7) // 2 + 1) // 2'
new = ('num_frames = c.num_frames if c.num_frames is not None '
       'else int(c.duration * 100)\n'
       '        T = ((num_frames - 7) // 2 + 1) // 2')
if old in src:
    p.write_text(src.replace(old, new))
    print('  Patched num_frames handling.')
else:
    print('  Already patched (or pattern moved).')
PY

# Symlink lhotse manifest dir into icefall's data path
mkdir -p data
if [[ ! -L data/fbank ]]; then
  ln -sfn /Users/skinnercheng/Projects/VOX_Personalis/out/lhotse_manifests data/fbank
fi

# Create case-clashing symlink that couldn't be created on macOS APFS
# (cuts_DEV.jsonl.gz vs cuts_dev.jsonl.gz collide on case-insensitive FS).
MANIFESTS=/Users/skinnercheng/Projects/VOX_Personalis/out/lhotse_manifests
ln -sfn cuts_dev.jsonl.gz "$MANIFESTS/cuts_DEV.jsonl.gz"

ls -la data/fbank/ | head -20

echo
echo "=== Verifying full stack ==="
python3 -c "
import torch, k2, lhotse
print(f'torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
print(f'k2:    {k2.__dev_version__}')
print(f'lhotse: {lhotse.__version__}')
"

echo
echo "================================================================"
echo "Bootstrap complete. To start the full fine-tune run:"
echo
echo "  cd ~/icefall/egs/librispeech/ASR"
echo "  python3 ./zipformer/finetune.py \\"
echo "      --do-finetune 1 \\"
echo "      --finetune-ckpt ~/base_ckpt_B/exp/pretrained.pt \\"
echo "      --bpe-model ~/base_ckpt_B/data/lang_bpe_500/bpe.model \\"
echo "      --num-epochs 20 --start-epoch 1 \\"
echo "      --use-fp16 1 --base-lr 0.0045 \\"
echo "      --causal 1 --chunk-size 32 --left-context-frames 128 \\"
echo "      --max-duration 1000 \\"
echo "      --enable-musan 0 \\"
echo "      --exp-dir ./zipformer/exp_finetune"
echo "================================================================"
