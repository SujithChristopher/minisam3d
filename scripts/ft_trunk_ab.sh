#!/bin/bash
# Quick A/B for the trunk fixes: fine-tune r18_v2_all for a few epochs with and without them, then score both
# (and the starting point) on the noark_v2 test split vs the teacher. The control arm gets the same extra
# training, so any gap is the change, not the epochs.
#   scripts/ft_trunk_ab.sh            (on the GPU box, in tmux; EPOCHS=n to change)
#   EPOCHS=3 PY=.venv/Scripts/python.exe TRAIN_ARGS="--map coco2014_vith=E:/NOARK_CV/msam_labels/coco2014_vith" \
#     scripts/ft_trunk_ab.sh          (Windows PC, Git Bash; TRAIN_ARGS also go to eval_teacher)
set -u
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-6}
PY=${PY:-.venv/bin/python}
BASE="--dataset noark_v2 --data-root data --image-dirs data --gray --rot --amp --backbone resnet18
      --init runs/r18_v2_all/best.pt --lr 3e-4 --epochs $EPOCHS --bs 64 --workers 8 ${TRAIN_ARGS:-}"
for arm in "ft_ctrl:" "ft_trunk:--trunk-w 3 --angle-sample 2"; do
  name=${arm%%:*} extra=${arm#*:}
  [ -f runs/$name/train.log ] && grep -q "^== ep $((EPOCHS - 1)) " runs/$name/train.log && continue
  mkdir -p runs/$name
  echo "[$(date +%T)] $name start: $extra"
  $PY -u scripts/train.py $BASE $extra --out runs/$name 2>&1 | tee runs/$name/train.log \
    | grep --line-buffered -E "^==|^dataset|^angle|Traceback|Error"
done
$PY scripts/eval_teacher.py --dataset noark_v2 --ckpt runs/r18_v2_all/best.pt runs/ft_ctrl/best.pt \
  runs/ft_trunk/best.pt ${TRAIN_ARGS:-} --out runs/ft_trunk_ab.json 2>&1 | tee runs/ft_trunk_ab.txt
echo "[$(date +%T)] AB DONE"
