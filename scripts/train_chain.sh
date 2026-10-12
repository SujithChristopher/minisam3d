#!/bin/bash
# Train a chain of models on one versioned dataset, one after another (for a tmux session on the GPU box).
#
#   scripts/train_chain.sh DATASET PREFIX FOLD [FOLD ...]
#   scripts/train_chain.sh noark_v2 r18_v2 all diwakar gokul js kabil ram
#
# FOLD "all" trains on the whole train split; any other FOLD is a subject left out (LOSO, grouped by person
# across sessions). Each fold writes runs/PREFIX_FOLD (run.json + checkpoints carry the dataset id). Re-running
# resumes an interrupted fold from its last.pt and skips finished ones. After "all", the test split is scored
# against the teacher (runs/PREFIX_all/test_eval.json). Defaults below are the ResNet-18 recipe; override
# with TRAIN_ARGS="..." (appended, so later flags win) or EPOCHS=n.
set -u
cd "$(dirname "$0")/.."
DATASET=$1 PREFIX=$2; shift 2
EPOCHS=${EPOCHS:-30}
PY=${PY:-.venv/bin/python}
BASE="--data-root data --image-dirs data --gray --rot --amp --backbone resnet18 --init runs/r18b_coco2014/best.pt
      --lr 1e-3 --epochs $EPOCHS --bs 64 --workers 8"
log() { echo "[$(date +%F\ %T)] $*" | tee -a runs/${PREFIX}_chain.log; }

$PY scripts/dataset.py verify "$DATASET" || { log "dataset $DATASET does not verify, stopping"; exit 1; }
for fold in "$@"; do
  out=runs/${PREFIX}_$fold
  mkdir -p "$out"
  if grep -q "^== ep $((EPOCHS - 1)) " "$out/train.log" 2>/dev/null; then log "$fold: done, skipping"; else
    excl=""; [ "$fold" != all ] && excl="--exclude $fold"
    resume=""; [ -f "$out/last.pt" ] && resume="--resume $out/last.pt"
    log "$fold: start ($DATASET $excl)"
    $PY -u scripts/train.py --dataset "$DATASET" $excl $BASE ${TRAIN_ARGS:-} --out "$out" $resume 2>&1 \
      | tee -a "$out/train.log" | grep --line-buffered -E "^==|^dataset|Traceback|Error"
    grep -q "^== ep $((EPOCHS - 1)) " "$out/train.log" || { log "$fold: FAILED, see $out/train.log"; continue; }
    log "$fold: finished"
  fi
  if [ "$fold" = all ] && [ ! -f "$out/test_eval.json" ]; then
    $PY scripts/eval_teacher.py --dataset "$DATASET" --ckpt ${COMPARE_CKPT:-} "$out/best.pt" \
      --out "$out/test_eval.json" 2>&1 | tee "$out/test_eval.txt"
    log "$fold: test split scored -> $out/test_eval.txt"
  fi
done
log "CHAIN DONE"
