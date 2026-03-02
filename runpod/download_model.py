#!/usr/bin/env python3
"""Download Step-Audio model weights into MODEL_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _has_weights(model_dir: str) -> bool:
    root = Path(model_dir)
    return bool(list(root.glob("*.safetensors")) or list(root.glob("*.bin")))


def main() -> None:
    model_id = os.getenv("MODEL_ID", "stepfun-ai/Step-Audio-R1.1")
    model_dir = os.getenv("MODEL_DIR", "/workspace/Step-Audio-R1.1")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(os.path.join(model_dir, "config.json")) and _has_weights(model_dir):
        print(f"Model already present at {model_dir}; skipping download.")
        return
    if os.path.exists(os.path.join(model_dir, "config.json")) and not _has_weights(model_dir):
        print(f"Model directory {model_dir} is missing weight shards; resuming download.")

    print(f"Downloading {model_id} into {model_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        token=hf_token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Model download completed.")


if __name__ == "__main__":
    main()
