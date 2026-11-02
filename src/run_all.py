"""Run A-D sequentially; stop immediately on an error."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def manifest(root, cfg):
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else (root / p).resolve()
    data = resolve(cfg["data_root"])
    model = resolve(cfg["checkpoint_dir"])
    sources = ["src/data.py", "src/model.py", "src/train.py", "src/run_all.py", "requirements.txt"]
    processor_files = ["config.json", "preprocessor_config.json", "tokenizer.json",
                       "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    return {"config": cfg,
            "source_sha256": {name: digest(root / name) for name in sources},
            "processor_sha256": {name: digest(model / name) for name in processor_files},
            "index_sha256": digest(data / "datasets" / "banana" / "index.jsonl"),
            "model_sha256": digest(model / "model.safetensors")}


def execute(args, root):
    """Run queued groups, retaining prior results on interruption."""

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_root = Path(cfg["output_root"])
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    frozen_config = output_root / "run_config.yaml"
    current = manifest(root, cfg)
    if current["model_sha256"] != cfg["checkpoint_safetensors_sha256"]:
        raise ValueError("Converted model hash does not match config")
    if args.resume:
        with manifest_path.open(encoding="utf-8") as f:
            previous = json.load(f)
        if previous != current:
            raise ValueError("Cannot resume: source, configuration, processor, data or model changed")
        with frozen_config.open(encoding="utf-8") as f:
            if yaml.safe_load(f) != cfg:
                raise ValueError("Frozen configuration differs from requested configuration")
    else:
        if manifest_path.exists() or frozen_config.exists():
            raise FileExistsError("Run manifest already exists; use --resume or a new output_root")
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        with frozen_config.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=True)
    for group in "ABCD":
        if manifest(root, cfg) != current:
            raise RuntimeError("Source, processor, dataset or model changed during the run; stopping")
        output = output_root / group
        if args.resume and (output / "test_metrics.json").is_file():
            print(f"SKIP_COMPLETED_GROUP {group}", flush=True)
            continue
        print(f"START_GROUP {group}", flush=True)
        command = [sys.executable, "-u", "-m", "src.train", "--config",
                   str(frozen_config), "--group", group]
        if args.resume and output.exists():
            if not (output / "last.pt").is_file():
                print(f"CANNOT_RESUME_GROUP {group}: no completed epoch checkpoint", flush=True)
                return 1
            command.append("--resume")
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            print(f"FAILED_GROUP {group} exit={result.returncode}", flush=True)
            return result.returncode
        print(f"COMPLETED_GROUP {group}", flush=True)
    print("ALL_GROUPS_COMPLETED", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    keep_awake = None
    if os.name == "nt":
        import ctypes
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        keep_awake.argtypes = [ctypes.c_ulong]
        keep_awake.restype = ctypes.c_ulong
        # Prevent idle sleep only while this runner lives; do not change the
        # machine's power plan, force the screen on, or block manual sleep.
        print(f"KEEP_AWAKE_ENABLED {bool(keep_awake(0x80000001))}", flush=True)
    try:
        return execute(args, root)
    finally:
        if keep_awake is not None:
            keep_awake(0x80000000)


if __name__ == "__main__":
    sys.exit(main())
