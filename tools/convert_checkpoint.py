"""One-time conversion of a verified local PyTorch checkpoint to safetensors.

Run with PyTorch >= 2.6 in a separate environment. Never load untrusted files.
"""
import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise RuntimeError("Conversion requires torch >= 2.6 due to CVE-2025-32434")
    source, destination = Path(args.input), Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != args.sha256.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual}")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("Unexpected checkpoint structure")
    save_file({k: v.contiguous() for k, v in state.items()}, str(destination), metadata={"format": "pt"})
    print(f"Saved {len(state)} tensors to {destination}", flush=True)


if __name__ == "__main__":
    main()
