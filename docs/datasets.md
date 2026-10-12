# Dataset versions

Every training set is a versioned manifest in `datasets/<name>.json`, checked into git. The data itself stays out
of git (`data/`). A manifest lists the label roots of each split, by path relative to `data/`, and pins every file
that defines them (`pack.bin`, `camera.npz`, `labels/*.npz`) by size and sha256. Its **id** is
`name@<8 hex of a hash over those pins and splits>`. If any byte or split membership changes, the id changes.

`scripts/train.py --dataset <name>` checks file sizes against the manifest before training. It then writes the
id, split, excluded subjects, exact roots and a code fingerprint into `<out>/run.json` and into every checkpoint
(`ck["dataset"]`, `ck["code"]`, `ck["git"]`). `ck["code"]` is a sha256 over `minisam3d/*.py` and
`scripts/train.py`. It is recorded because code reaches the GPU box by copying, so its git checkout can be stale.

| version | train | test | frames (train / test) |
|---|---|---|---|
| `noark_v1@23acecce` | COCO 2014 teacher labels + sep23 game/woc, 5 subjects | none | 35,793 / 0 |
| `noark_v2@65c0cf7c` | v1 + sep23 `_wc` + oct takes of diwakar, gokul, js, ram | oct arjun, jerome, lee (9 takes) | 67,013 / 19,305 |

COCO images (`data/images`) are not pinned, only their label shards. `v3rot_loso_*` and `r18b_loso_*` predate
versioning. Their `run.json` is back-filled as `noark_v1` (same takes and file sizes, not hashed at training time).

## Commands

```bash
# new version (a changed take needs a new name; build refuses to overwrite a different id)
python scripts/dataset.py build noark_v3 --train coco2014_vith 'noark_pack/*' ... --test ... --note "..." \
       --map coco2014_vith=E:/NOARK_CV/msam_labels/coco2014_vith      # roots outside data/
python scripts/dataset.py verify noark_v2 [--full]                    # sizes; --full also sha256
MSAM_SSH_PASSWORD=... python scripts/dataset.py push noark_v2 --remote parvathy@172.17.13.173:minisam3d
python scripts/dataset.py runs                                        # every run and what it trained on

# train / evaluate
python scripts/train.py --dataset noark_v2 --exclude ram ...          # LOSO: groups sep23 + oct by person
scripts/train_chain.sh noark_v2 r18_v2 all diwakar gokul js kabil ram # on the GPU box, in tmux
python scripts/eval_teacher.py --dataset noark_v2 --ckpt runs/a/best.pt runs/b/best.pt   # test split vs teacher
```

`push` sends only roots whose remote file sizes differ, streamed as tar. It then checks sha256 on the remote and
copies the manifest. The password is read from `$MSAM_SSH_PASSWORD` or a prompt and is never written anywhere.
