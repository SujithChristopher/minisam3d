"""Versioned training datasets: a manifest per version, checked into git under datasets/.

A manifest names the label roots (pack folders, or a labels-only root like coco2014_vith) of each split, by
path relative to a data root, and pins every file that defines the training data (pack.bin, camera.npz,
labels/*.npz) by size and sha256. Its id (name@first 8 hex of a hash over those pins) is what train.py writes
into run.json and every checkpoint, so a model can always be traced to the exact bytes it saw.

  python scripts/dataset.py build noark_v2 --train noark_pack/* noark_pack_oct/s3_diwakar_* ... \
                                           --test noark_pack_oct/s3_arjun_* ... --note "..."
  python scripts/dataset.py verify noark_v2 [--root data] [--full]   sizes always, hashes with --full
  python scripts/dataset.py push noark_v2 --remote parvathy@172.17.13.173:minisam3d
  python scripts/dataset.py runs                                    which run trained on what

Labels-only roots that live outside the data root are mapped with --map NAME=PATH (build and verify), e.g.
--map coco2014_vith=E:/NOARK_CV/msam_labels/coco2014_vith. push copies only roots it finds under --root and
whose remote copy differs in size; it logs in with the key of an ~/.ssh/config alias (gpubox), else the password
from $MSAM_SSH_PASSWORD or a prompt, never a file.
"""
import argparse
import datetime
import getpass
import glob
import hashlib
import json
import os
import posixpath
import sys
import tarfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DS_DIR = os.path.join(ROOT, "datasets")


def pinned_files(d):
    """The files that define a label root's data (not logs, not linked images)."""
    fs = [f for f in ("pack.bin", "camera.npz") if os.path.isfile(os.path.join(d, f))]
    fs += sorted(posixpath.join("labels", os.path.basename(p)) for p in glob.glob(os.path.join(d, "labels", "*.npz")))
    return fs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def frames(d):
    import numpy as np
    return sum(len(np.load(p, allow_pickle=True)["image"]) for p in glob.glob(os.path.join(d, "labels", "*.npz")))


def manifest_id(m):
    pins = {k: v["files"] for k, v in sorted(m["roots"].items())}
    splits = {k: sorted(v) for k, v in sorted(m["splits"].items())}
    blob = json.dumps(dict(pins=pins, splits=splits), sort_keys=True).encode()
    return f"{m['name']}@{hashlib.sha256(blob).hexdigest()[:8]}"


def load(name):
    path = name if name.endswith(".json") else os.path.join(DS_DIR, f"{name}.json")
    return json.load(open(path, encoding="utf-8"))


def local_dir(rel, root, maps):
    return maps.get(rel, os.path.join(root, rel))


def parse_maps(specs):
    return dict(s.split("=", 1) for s in specs or [])


def expand(patterns, root, maps):
    out = []
    for p in patterns:
        if p in maps:
            out.append(p)
            continue
        hits = sorted(os.path.relpath(h, root).replace(os.sep, "/") for h in glob.glob(os.path.join(root, p))
                      if os.path.isdir(os.path.join(h, "labels")))
        if not hits:
            sys.exit(f"no label roots match {p} under {root}")
        out += hits
    return list(dict.fromkeys(out))


def describe(rel):
    """subject / session / kind from the take folder name (s3_gokul_wc -> gokul, oct, wc)."""
    base = posixpath.basename(rel)
    if "_" not in base or base.startswith("coco"):
        return {}
    session = "oct" if base.startswith("s3_") else "sep23"
    subj, kind = base.removeprefix("s3_").rsplit("_", 1)
    return dict(subject=subj, session=session, kind=kind)


def cmd_build(a):
    maps = parse_maps(a.map)
    splits = {"train": expand(a.train, a.root, maps)}
    if a.test:
        splits["test"] = expand(a.test, a.root, maps)
    overlap = set(splits["train"]) & set(splits.get("test", []))
    if overlap:
        sys.exit(f"in both train and test: {sorted(overlap)}")
    roots = {}
    for rel in sorted({r for v in splits.values() for r in v}):
        d = local_dir(rel, a.root, maps)
        files = {}
        for f in pinned_files(d):
            p = os.path.join(d, f)
            files[f] = dict(bytes=os.path.getsize(p), sha256=sha256(p))
        roots[rel] = dict(frames=frames(d), files=files, **describe(rel))
        print(f"  {rel:<28} {roots[rel]['frames']:>7} frames  {sum(v['bytes'] for v in files.values()) / 2**20:8.0f} MB")
    m = dict(name=a.name, created=datetime.date.today().isoformat(), note=a.note, splits=splits, roots=roots)
    m["id"] = manifest_id(m)
    os.makedirs(DS_DIR, exist_ok=True)
    out = os.path.join(DS_DIR, f"{a.name}.json")
    if os.path.exists(out) and not a.force:
        old = json.load(open(out, encoding="utf-8"))
        if old.get("id") != m["id"]:
            sys.exit(f"{out} exists with different content ({old.get('id')} vs {m['id']}); "
                     "make a new version name instead (or --force)")
    json.dump(m, open(out, "w", encoding="utf-8"), indent=1)
    for k, v in splits.items():
        print(f"{k}: {len(v)} roots, {sum(roots[r]['frames'] for r in v)} frames")
    print(f"wrote {out}  id {m['id']}")


def verify(m, root, maps=None, full=False, roots=None):
    """Problems (empty list = ok) between the manifest and the files on disk (all roots, or only `roots`)."""
    maps, bad = maps or {}, []
    if manifest_id(m) != m.get("id"):
        bad.append(f"manifest edited by hand: id {m.get('id')} but content hashes to {manifest_id(m)}")
    want = set(roots) if roots is not None else {r for v in m["splits"].values() for r in v}
    for rel in sorted(want):
        d = local_dir(rel, root, maps)
        for f, pin in m["roots"][rel]["files"].items():
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                bad.append(f"{rel}/{f}: missing")
            elif os.path.getsize(p) != pin["bytes"]:
                bad.append(f"{rel}/{f}: {os.path.getsize(p)} bytes, manifest {pin['bytes']}")
            elif full and sha256(p) != pin["sha256"]:
                bad.append(f"{rel}/{f}: sha256 differs")
        extra = set(pinned_files(d)) - set(m["roots"][rel]["files"]) if os.path.isdir(d) else set()
        bad += [f"{rel}/{f}: not in the manifest" for f in sorted(extra)]
    return bad


def cmd_verify(a):
    m = load(a.name)
    bad = verify(m, a.root, parse_maps(a.map), a.full)
    print("\n".join(bad) or f"{m['id']}: ok ({'sha256' if a.full else 'sizes'})")
    sys.exit(1 if bad else 0)


def cmd_push(a):
    import paramiko
    m = load(a.name)
    host, rdir = a.remote.split(":", 1)
    user, host = host.split("@", 1) if "@" in host else (None, host)
    # an ~/.ssh/config alias (e.g. gpubox) supplies host, user and key; the password is only the fallback
    cfg_path = os.path.expanduser("~/.ssh/config")
    cfg = paramiko.SSHConfig.from_path(cfg_path).lookup(host) if os.path.exists(cfg_path) else {}
    host, user = cfg.get("hostname", host), user or cfg.get("user")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, username=user, key_filename=[os.path.expanduser(k) for k in cfg.get("identityfile", [])],
                  timeout=15, allow_agent=False, look_for_keys=not cfg.get("identityfile"))
    except paramiko.AuthenticationException:
        pw = os.environ.get("MSAM_SSH_PASSWORD") or getpass.getpass(f"{user}@{host} password: ")
        c.connect(host, username=user, password=pw, timeout=15, allow_agent=False, look_for_keys=False)
    c.get_transport().window_size = 2 ** 27

    def sh(cmd):
        _, o, e = c.exec_command(cmd)
        rc = o.channel.recv_exit_status()
        return rc, o.read().decode(errors="replace"), e.read().decode(errors="replace")

    rdata = posixpath.join(rdir, a.remote_data)
    todo = []
    for rel, info in sorted(m["roots"].items()):
        if not os.path.isdir(os.path.join(a.root, rel)):
            print(f"  {rel}: not under {a.root}, expecting it on the remote already")
            continue
        _, out, _ = sh("cd %s 2>/dev/null && stat -c '%%n %%s' %s 2>/dev/null"
                       % (posixpath.join(rdata, rel), " ".join(info["files"])))
        have = dict(line.rsplit(" ", 1) for line in out.splitlines())
        if all(have.get(f) == str(p["bytes"]) for f, p in info["files"].items()):
            continue
        todo.append(rel)
    total = sum(sum(p["bytes"] for p in m["roots"][r]["files"].values()) for r in todo)
    print(f"{len(todo)} roots to send ({total / 2**30:.2f} GB): {todo}")
    if todo:
        ch = c.get_transport().open_session()
        ch.exec_command(f"mkdir -p {rdata} && tar -x -C {rdata}")
        sent, t0 = [0], time.time()

        class Pipe:
            def write(self, b):
                ch.sendall(b)
                n0, sent[0] = sent[0], sent[0] + len(b)
                if n0 >> 28 != sent[0] >> 28:  # every 256 MB
                    print(f"  {sent[0] / 2**30:.2f} GB  {sent[0] / 2**20 / (time.time() - t0):.0f} MB/s", flush=True)
                return len(b)

        with tarfile.open(fileobj=Pipe(), mode="w|", bufsize=1 << 20) as tf:
            for rel in todo:
                for f in m["roots"][rel]["files"]:
                    tf.add(os.path.join(a.root, rel, f), arcname=posixpath.join(rel, f))
        ch.shutdown_write()
        if ch.recv_exit_status():
            sys.exit(f"remote tar failed: {ch.makefile_stderr().read().decode(errors='replace')}")
    # manifest next to the code, then check the remote bytes against it
    sh(f"mkdir -p {posixpath.join(rdir, 'datasets')}")
    sf = c.open_sftp()
    sf.put(os.path.join(DS_DIR, f"{m['name']}.json"), posixpath.join(rdir, "datasets", f"{m['name']}.json"))
    lines = [f"{p['sha256']}  {posixpath.join(rel, f)}" for rel in todo for f, p in m["roots"][rel]["files"].items()]
    if lines:
        sf.open(posixpath.join(rdata, ".push.sha256"), "w").write("\n".join(lines) + "\n")
        rc, out, err = sh(f"cd {rdata} && sha256sum --quiet -c .push.sha256; rc=$?; rm -f .push.sha256; exit $rc")
        if rc:
            sys.exit(f"sha256 mismatch on the remote:\n{out}{err}")
    print(f"{m['id']}: remote in sync ({len(todo)} sent, sha256 checked)")


def cmd_runs(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "run.json"))):
        r = json.load(open(p, encoding="utf-8"))
        ds = r.get("dataset") or {}
        rows.append((os.path.basename(os.path.dirname(p)), ds.get("id", "-"), ds.get("split", "-"),
                     ",".join(ds.get("exclude", [])) or "-", r.get("args", {}).get("backbone", "-"),
                     r.get("code", "-"), r.get("started", "-")))
    print(f"{'run':<26} {'dataset':<20} {'split':<6} {'excluded':<14} {'backbone':<9} {'code':<12} started")
    for row in rows:
        print("{:<26} {:<20} {:<6} {:<14} {:<9} {:<12} {}".format(*row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--train", nargs="+", required=True, help="label roots or globs, relative to --root")
    b.add_argument("--test", nargs="*", help="held out, never trained on")
    b.add_argument("--note", default="")
    b.add_argument("--force", action="store_true", help="overwrite an existing version with new content")
    v = sub.add_parser("verify")
    v.add_argument("name")
    v.add_argument("--full", action="store_true", help="also compare sha256 (reads every byte)")
    for s in (b, v):
        s.add_argument("--root", default=os.path.join(ROOT, "data"))
        s.add_argument("--map", nargs="*", help="NAME=PATH for roots outside --root")
    p = sub.add_parser("push")
    p.add_argument("name")
    p.add_argument("--remote", required=True, help="user@host:repo_dir (repo_dir relative to home, or absolute)")
    p.add_argument("--remote-data", default="data", help="data root inside repo_dir")
    p.add_argument("--root", default=os.path.join(ROOT, "data"))
    sub.add_parser("runs")
    a = ap.parse_args()
    dict(build=cmd_build, verify=cmd_verify, push=cmd_push, runs=cmd_runs)[a.cmd](a)


if __name__ == "__main__":
    main()
